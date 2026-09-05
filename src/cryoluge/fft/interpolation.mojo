
from math import floor, ceil, ceildiv
from complex import ComplexSIMD
from utils.numerics import inf, nan, isinf, isnan
from os import abort

from cryoluge.collections import MovableList
from cryoluge.math import Vec, AlignedBox, OrientedBox, complex, ladder, round
from cryoluge.image import DimensionalBuffer
from cryoluge.image.analysis import FrequencyLimits, FrequencyLimitsChecker
from cryoluge.fft import FFTCoordsFull, Delta


comptime SIMDInt[simd_width: Int] = SIMD[DType.int,simd_width]
comptime SIMDBool[simd_width: Int] = SIMD[DType.bool,simd_width]


@fieldwise_init
struct OutOfRangeBehavior[dtype: DType](
    Movable,
    ImplicitlyCopyable,
    Writable,
    Stringable
):
    var id: Int
    var value: ComplexScalar[dtype]

    alias Interpolate: Int = 1
    alias Override: Int = 2

    @staticmethod
    fn interpolate(v: ComplexScalar[dtype], out s: Self):
        s = Self(Self.Interpolate, v)

    @staticmethod
    fn override(v: ComplexScalar[dtype], out s: Self):
        s = Self(Self.Override, v)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("OutOfRangeBehavior[")
        if self.id == Self.Interpolate:
            writer.write("interpolate")
        elif self.id == Self.Override:
            writer.write("override")
        else:
            writer.write("(unknown)")
        writer.write(", v=", self.value, "]")

    fn __str__(self) -> String:
        return String.write(self)


struct PrecomputedFFTInterpolationFull[
    dim: Int,
    dtype: DType,
    out_of_range: OutOfRangeBehavior[dtype],
    *,
    dtype_coords: DType = dtype,
](Movable):
    """
    A SIMD-optimized implementation of multi-dimensional linear interpolation
    that pre-computes a cache of sampled pixel neighborhoods to maximize memory locality.
    WARNING: uses ~16x more memory than the original image data,
             which will overflow CPU caches much more quickly with increasing image sizes!
             Profiling generally shows the overall performance improvement is worth it,
             although the gains start to diminish with increasing image size.
    """
    var _sizes_real: Vec[dim,Int]
    var _samples: DimensionalBuffer[dim,Self.Pixel]

    comptime deltas = Delta[dim,dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[dtype,Self.num_samples]
    comptime empty_samples = ComplexSIMD[dtype,Self.num_samples](
        re=Self.out_of_range.value.re,
        im=Self.out_of_range.value.im
    )
    comptime Selector = _Selector[Self.num_samples]

    fn __init__(
        out self,
        img: FFTImage[dim,dtype]
    ):

        self._sizes_real = img.sizes_real.copy()

        # allocate storage for all the samples
        # we'll explicitly represent the other half of the x dimension to avoid complexity due to conjugation
        var sizes = FFTCoordsFull(self._sizes_real).sizes_fourier() + 1
        # NOTE: need an extra pixel in each dimension to interpolate before fmin
        self._samples = DimensionalBuffer[dim,Self.Pixel](sizes)

        # precompute all the pixel samples
        @parameter
        fn func(i: Vec[dim,Int]):

            var f = self._i2f(i)
            var pixel = Self.Pixel(0, 0)

            @parameter
            for s in range(Self.num_samples):

                # sample the point
                var f_sample = f + materialize[Self.deltas[s].pos]()
                var v = img.find(f=f_sample)

                # handle out-of-range behavior
                if v is None:
                    @parameter
                    if out_of_range.id == OutOfRangeBehavior.Interpolate:
                        # interpolate with the out-of-range value
                        pixel.re[s] = out_of_range.value.re
                        pixel.im[s] = out_of_range.value.im
                    elif out_of_range.id == OutOfRangeBehavior.Override:
                        # override the whole pixel with the out-of-range value
                        pixel.re = SIMD[dtype,Self.num_samples](out_of_range.value.re)
                        pixel.im = SIMD[dtype,Self.num_samples](out_of_range.value.im)
                        break
                else:
                    # otherwise, just interpolate with the sampled value like normal
                    pixel.re[s] = v.value().re
                    pixel.im[s] = v.value().im

            self._samples[i=i] = pixel
                
        self._samples.iterate[func]()

    fn _coords(self) -> FFTCoordsFull[dim]:
        return FFTCoordsFull(self._sizes_real)

    @always_inline
    fn _offset[d: Int](self, out offset: Int):
        offset = (self._sizes_real[d] + 2) >> 1

    @always_inline
    fn _imax[d: Int](self, out imax: Int):
        imax = self._sizes_real[d]
        @parameter
        if d == 0:
            imax |= 0b1

    @always_inline
    fn _i2f(self, i: Vec[dim,Int], out f: Vec[dim,Int]):
        """
        Maps interpolation storage coordinates into frequency coordinates.
        NOTE: This is not the same transformation as FFTCoords.i2f(),
              since the storage layouts are different.
        """

        f = Vec[dim,Int](uninitialized=True)

        @parameter
        for d in range(0, dim):
            f[d] = i[d] - self._offset[d]()
        
    @always_inline
    fn _f2i[simd_width: Int](
        self,
        f: Vec[dim,SIMDInt[simd_width]],
        out i: Vec[dim,SIMDInt[simd_width]]
    ):
        """
        Maps frequency coordinates into the interpolation storage coordinates.
        NOTE: This is not the same transformation as FFTCoords.f2i(),
              since the storage layouts are different.
        """
        
        i = Vec[dim,SIMDInt[simd_width]](uninitialized=True)

        @parameter
        for d in range(0, dim):
            i[d] = f[d] + self._offset[d]()

            # if out of range, replace with -1
            var out_of_range = i[d].lt(0) or i[d].gt(self._imax[d]())
            i[d] = out_of_range.select(
                true_case = SIMDInt[simd_width](-1),
                false_case = i[d]
            )

    fn _start_dists[
        simd_width: Int
    ](
        self,
        f: Vec[dim,SIMD[dtype_coords,simd_width]],
        out result: Tuple[
            Vec[dim,SIMDInt[simd_width]],
            Vec[dim,SIMD[dtype_coords,simd_width]]
        ]
    ):

        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[dim,SIMDInt[simd_width]](uninitialized=True)
        var dists = Vec[dim,SIMD[dtype_coords,simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var floor = floor(f[d])
            start[d] = SIMDInt[simd_width](floor)
            dists[d] = f[d] - floor

        result = (start^, dists^)

    fn _neighborhood(
        self,
        *,
        i: Vec[dim,Int],
        out v: ComplexSIMD[dtype,Self.num_samples]
    ):
        v = self._samples.get(i)
            .or_else(Self.empty_samples)

    fn get[simd_width: Int](
        self,
        *,
        f: Vec[dim,SIMD[dtype_coords,simd_width]],
        out v: ComplexSIMD[dtype,simd_width]
    ):
        var result = self._start_dists(f)
        ref start = result[0]
        ref dists = result[1]

        var i = self._f2i(start)

        v = ComplexSIMD[dtype,simd_width](re=0, im=0)

        @parameter
        for w in range(simd_width):
            var neighborhood = self._neighborhood(i=i[slice=w].map_int())
            var vw = interpolate(dists[slice=w], neighborhood)
            v.re[w] = vw.re
            v.im[w] = vw.im


fn interpolate[
    dim: Int,
    dtype: DType,
    dtype_coords: DType,
    num_samples: Int
](
    dists: Vec[dim,Scalar[dtype_coords]],
    var samples: ComplexSIMD[dtype,num_samples],
    out v: ComplexScalar[dtype]
):
    v = ComplexScalar[dtype](re=0, im=0)

    # apply sample weights based on the distances
    @parameter
    for d in range(dim):
        var t = SIMD[dtype,num_samples](dists[d])
        var omt = SIMD[dtype,num_samples](1 - dists[d])
        comptime selector = _make_selector[dim,num_samples](d)
        var w = selector.select(omt, t)
        samples.re *= w
        samples.im *= w

    # the final interpolated pixel is the sum of the weighted samples
    v.re = samples.re.reduce_add()
    v.im = samples.im.reduce_add()


struct PrecomputedFFTInterpolationNop[
    dim: Int,
    dtype: DType,
    out_of_range: OutOfRangeBehavior[dtype],
    *,
    dtype_coords: DType = dtype
](Movable):
    """
    A no-op implementation of the FFT interpolation, for testing,
    to see how well (or poorly) doing the interpolation with incoherent memory accesses really is.
    NOTE: It's very poor.
    """
    var _img: FFTImage[dim,dtype]

    comptime deltas = Delta[dim,dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[dtype,Self.num_samples]
    comptime EmptySamples[c: ComplexSIMD[dtype,1]] = ComplexSIMD[dtype,Self.num_samples](
        re=SIMD[dtype,Self.num_samples](c.re),
        im=SIMD[dtype,Self.num_samples](c.im)
    )

    fn __init__(
        out self,
        img: FFTImage[dim,dtype]
    ):
        self._img = img.copy()

    fn get[simd_width: Int](
        self,
        *,
        f: Vec[dim,SIMD[dtype_coords,simd_width]],
        out v: ComplexSIMD[dtype,simd_width]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[dim,SIMDInt[simd_width]](uninitialized=True)
        var dists = Vec[dim,SIMD[dtype_coords,simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var floor = floor(f[d])
            start[d] = SIMDInt[simd_width](floor)
            dists[d] = f[d] - floor

        v = ComplexSIMD[dtype,simd_width](re=0, im=0)

        @parameter
        for w in range(simd_width):

            # load the samples
            # NOTE: this part just loads all 2,4, or 8 pixels independently,
            #       hoping that limited locality in the x-dimension will give somewhat good cache performance
            var samples = Self.Pixel(0, 0)
            @parameter
            for s in range(Self.num_samples):
                var f_sample = start[slice=w].map_int() + materialize[Self.deltas[s].pos]()
                var v = self._img.find(f=f_sample)

                # handle out-of-range behavior
                if v is None:
                    @parameter
                    if out_of_range.id == OutOfRangeBehavior.Interpolate:
                        # interpolate with the out-of-range value
                        samples.re[s] = out_of_range.value.re
                        samples.im[s] = out_of_range.value.im
                    elif out_of_range.id == OutOfRangeBehavior.Override:
                        # override the whole pixel with the out-of-range value
                        samples.re = SIMD[dtype,Self.num_samples](out_of_range.value.re)
                        samples.im = SIMD[dtype,Self.num_samples](out_of_range.value.im)
                        break
                else:
                    # otherwise, just interpolate with the sampled value like normal
                    samples.re[s] = v.value().re
                    samples.im[s] = v.value().im

            var vw = interpolate(dists[slice=w], samples)
            v.re[w] = vw.re
            v.im[w] = vw.im


comptime _Selector[num_samples: Int] = SIMDBool[num_samples]

fn _make_selector[
    dim: Int,
    num_samples: Int
](d: Int, out selector: _Selector[num_samples]):
    
    comptime S = _Selector[num_samples]
    comptime t = False
    comptime omt = True
    var s0 = SIMDBool[2](omt, t)

    @parameter
    if dim == 1:
        selector = rebind[S](s0)
    elif dim == 2:
        if d == 0:selector = rebind[S](s0.join(s0))
        elif d == 1:
            selector = rebind[S](s0.interleave(s0))
        else:
            selector = abort[S]("d exceeds rank 2")
    elif dim == 3:
        if d == 0:
            var s1 = s0.join(s0)
            selector = rebind[S](s1.join(s1))
        elif d == 1:
            var s1 = s0.interleave(s0)
            selector = rebind[S](s1.join(s1))
        elif d == 2:
            var s1 = s0.interleave(s0)
            selector = rebind[S](s1.interleave(s1))
        else:
            selector = abort[S]("d exceeds rank 3")
    else:
        constrained[False, String("unrecognized dimension: ", dim)]()
        selector = abort[S]()


comptime PrecomputedFFTInterpolation = PrecomputedFFTInterpolationFull
# NOTE: this is useful for switching downstream apps to use different implementations during benchmarking


struct VolumeNeighborhoods[
    dtype: DType,
    simd_width: Int,
    out_of_range: OutOfRangeBehavior[dtype],
    *,
    dtype_coords: DType = dtype
](Movable):
    var _sizes_real_vol: Vec[3,Int]
    var _segments: DimensionalBuffer[3,Self.Segment]

    comptime Segment = ComplexSIMD[dtype,simd_width]
    comptime out_of_range_segment = Self.Segment(
        re=out_of_range.value.re,
        im=out_of_range.value.im
    )
    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()

    fn __init__(
        out self,
        img: FFTImage[3,dtype]
    ):
        self._sizes_real_vol = img.sizes_real.copy()

        # calculate how many segments we need in each x-row
        var coords = FFTCoords(self._sizes_real_vol)
        var sizes_fourier = coords.sizes_fourier()
        var sizes_segments = sizes_fourier.copy()
        var sizes_segments.x() = ceildiv(sizes_fourier.x(), Self.num_neighborhoods_in_segment)
        
        # allocate storage for all the segments
        self._segments = DimensionalBuffer[3,Self.Segment](sizes_segments)

        # pack all the segments
        @parameter
        fn func(s: Vec[3,Int]):

            var segment = Self.out_of_range_segment

            # convert segment indices into image indices
            var i = s.copy()
            i.x() = s.x()*Self.num_neighborhoods_in_segment

            var f = coords.i2f_contiguous(i=i)

            # pack all the pixels into this segment
            @parameter
            for w in range(Self.simd_width):

                # if the pixel is inside the volume, pack it
                # (otherwise, leave it out-of-range)
                var fw = f + Vec[3](x=w, y=0, z=0)
                var iw = coords.maybe_f2i(fw)
                if iw is not None:
                    complex.splice(segment, w, img.complex[i=iw.value()])

            self._segments[i=s] = segment
                
        sizes_segments.iterate_over_sizes[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords

    @always_inline
    fn coords(self) -> FFTCoords[3]:
        return FFTCoords(self._sizes_real_vol)
    
    @always_inline
    fn _segment(
        self,
        *,
        i: Vec[3,Int],
        out segment: Self.Segment
    ):
        # map to segment indices
        var s = i.copy()
        s.x() //= Self.num_neighborhoods_in_segment

        segment = self._segments[i=s]

    @always_inline
    fn _segment(
        self,
        *,
        i: Optional[Vec[3,Int]],
        out segment: Self.Segment
    ):
        if i is not None:
            segment = self._segment(i=i.value())
        else:
            segment = Self.out_of_range_segment

    # TODO: @always_inline ?
    fn _segment_neighborhood[x_halfspace: Int](
        self,
        f_vi_pos: Vec[3,Int],
        out segment_neighborhood: _SegmentNeighborhood[dtype,simd_width]
    ):
        # a convoluted example on a 6x6 image (same thing for 7x7 image):
        # neighborhood at f_vi_pos=0,1  x_halfspace=1 :
        #     -3 -2 -1    0  1  2  3
        # +2           | 05 15 25 35  +2
        # +1           | 04 14 24 34  +1
        #              | n0 n1 n2 --
        # neighborhood at f_vi_pos=0,1  x_halfspace=-1 :
        #     -3 -2 -1    0  1  2  3
        # -1  34 24 14 | 02           -1
        # -2  35 25 15 | 01           -2
        #     n2 n1 n0   --
        # need to replace the x=0 column with values from another neighborhood:
        #     -3 -2 -1    0  1  2  3
        # -1           | 02 12 22 32  -1
        # -2           | 01 11 21 31  -2

        # another example on a 6x6 image:
        # neighborhood at f_vi_pos=0,-3  x_halfspace=1 :
        # -2           | 01 11 21 31  -2
        # -3           | 00 10 20 30  -3
        #                n0 n1 n2 --
        # neighborhood at f_vi_pos=0,1  x_halfspace=-1 :
        #     -3 -2 -1    0  1  2  3
        # +3  OR OR OR | OR
        # +2  31 21 11 | 05           +2
        #     n2 n1 n0   --

        var f_vi = f_vi_pos.copy()

        # handle the -x halfspace here by inverting the coordinates and directions both,
        # so we don't need to subtract 1 from the coordinates
        @parameter
        if x_halfspace == -1:
            f_vi *= -1

        comptime f_dy = Vec[3](x=0, y=1, z=0)*x_halfspace
        comptime f_dz = Vec[3](x=0, y=0, z=1)*x_halfspace
        comptime f_d00 = Vec[3](x=0, y=0, z=0)
        comptime f_d10 = f_dy
        comptime f_d01 = f_dz
        comptime f_d11 = f_dy + f_dz

        var coords = self.coords()

        # get the neighborhood image coordinates
        var i_vi_00 = coords.maybe_f2i_contiguous(f_vi + materialize[f_d00]())
        var i_vi_10 = coords.maybe_f2i_contiguous(f_vi + materialize[f_d10]())
        var i_vi_01 = coords.maybe_f2i_contiguous(f_vi + materialize[f_d01]())
        var i_vi_11 = coords.maybe_f2i_contiguous(f_vi + materialize[f_d11]())

        # apply out-of-range behvavior
        var in_range_00 = i_vi_00 is not None  # TODO: always true?
        var in_range_10 = i_vi_10 is not None
        var in_range_01 = i_vi_01 is not None
        var in_range_11 = i_vi_11 is not None  # TODO: always 10 or 01 ?
        var in_range_yz = in_range_10 and in_range_01  # TODO: redundant?
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not in_range_yz:
                i_vi_00 = None
                i_vi_10 = None
                i_vi_01 = None
                i_vi_11 = None

        # set the x-in-range mask
        var fx_vi_segment = materialize[ladder[simd_width]()]() + f_vi_pos.x()
        var in_range_x_mask = fx_vi_segment.le(coords.fmax[0]())

        # apply out-of-range behavior
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not in_range_yz:
                in_range_x_mask = SIMDBool[simd_width](fill=False)

        # read the segments, where possible
        segment_neighborhood = _SegmentNeighborhood[dtype,simd_width](
            s00 = self._segment(i=i_vi_00),
            s10 = self._segment(i=i_vi_10),
            s01 = self._segment(i=i_vi_01),
            s11 = self._segment(i=i_vi_11),
            in_range_x_mask = in_range_x_mask
        )

        @parameter
        if x_halfspace == -1:

            # x = 0 doesn't have x-contiguous voxels,
            # so we need to load the missing voxels from different segments
            if f_vi.x() == 0:

                # we're going to conjugate all the value in the neighborhood later on,
                # but the un-patched values shouldn't be conjugated because they're from the +x halfspace.
                # so conjugate them now, so the later conjugation puts them back to normal,
                # but only if they're in-range
                if in_range_x_mask[0]:
                    if in_range_00:
                        segment_neighborhood.s00.im[0] *= -1
                    if in_range_10:
                        segment_neighborhood.s10.im[0] *= -1
                    if in_range_01:
                        segment_neighborhood.s01.im[0] *= -1
                    if in_range_11:
                        segment_neighborhood.s11.im[0] *= -1
                # TODO: can we simplify this?

                @always_inline
                fn flipped_i[f_d: Vec[3,Int]](
                    coords: FFTCoords[3],
                    f_vi: Vec[3,Int],
                    in_range: Bool,
                    out i: Optional[Vec[3,Int]]
                ):
                    # keep the same coordinate boundaries as before
                    if not in_range:
                        i = None
                        return

                    var f = f_vi + materialize[f_d]()

                    @parameter
                    for d in range(1, 3):
                        f[d] *= -1
                        if f[d] > coords.fmax[d]():
                            f[d] -= coords.size_fourier[d]()

                    i = coords.f2i_contiguous(f)
                    # TODO: can this be simplified/optimized at all?

                # get the extra voxel image coordinates
                i_vi_00 = flipped_i[f_d00](coords, f_vi, in_range_00)
                i_vi_10 = flipped_i[f_d10](coords, f_vi, in_range_10)
                i_vi_01 = flipped_i[f_d01](coords, f_vi, in_range_01)
                i_vi_11 = flipped_i[f_d11](coords, f_vi, in_range_11)

                # apply out-of-range behavior
                @parameter
                if out_of_range.id == OutOfRangeBehavior.Override:
                    if not in_range_yz:
                        i_vi_00 = None
                        i_vi_10 = None
                        i_vi_01 = None
                        i_vi_11 = None

                # replace the affected voxels
                complex.splice[Self.num_neighborhoods_in_segment](segment_neighborhood.s00, 1, self._segment(i=i_vi_00), 1)
                complex.splice[Self.num_neighborhoods_in_segment](segment_neighborhood.s10, 1, self._segment(i=i_vi_10), 1)
                complex.splice[Self.num_neighborhoods_in_segment](segment_neighborhood.s01, 1, self._segment(i=i_vi_01), 1)
                complex.splice[Self.num_neighborhoods_in_segment](segment_neighborhood.s11, 1, self._segment(i=i_vi_11), 1)
                # TODO: try splicing one value into the newly-loaded segment, then overwrite the neighborhood

            # conjugate all the in-range voxels
            @always_inline
            fn conj_mask(
                in_range_x_mask: SIMDBool[simd_width],
                in_range: Bool,
                out conj_mask: SIMD[dtype,simd_width]
            ):
                var in_range_mask = SIMDBool[simd_width](fill=in_range)
                # NOTE: `a and b` doesn't do the vectorized boolean operation here,
                #       (due to implicit conversions?)
                #       so we need `a.__and__(b)`
                conj_mask = (in_range_x_mask.__and__(in_range_mask)).select(
                    true_case = Scalar[dtype](-1),
                    false_case = Scalar[dtype](1)
                )
            
            segment_neighborhood.s00.im *= conj_mask(in_range_x_mask, in_range_00)
            segment_neighborhood.s10.im *= conj_mask(in_range_x_mask, in_range_10)
            segment_neighborhood.s01.im *= conj_mask(in_range_x_mask, in_range_01)
            segment_neighborhood.s11.im *= conj_mask(in_range_x_mask, in_range_11)

        # apply post-conjugation out-of-range behavior
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            segment_neighborhood.in_range_x_mask = segment_neighborhood.in_range_x_mask.shift_left[1]()
            segment_neighborhood.in_range_x_mask[simd_width - 1] = False
        
        @parameter
        if x_halfspace == -1:

            # reverse the voxel order
            segment_neighborhood.s00.re = segment_neighborhood.s00.re.reversed()
            segment_neighborhood.s00.im = segment_neighborhood.s00.im.reversed()
            segment_neighborhood.s10.re = segment_neighborhood.s10.re.reversed()
            segment_neighborhood.s10.im = segment_neighborhood.s10.im.reversed()
            segment_neighborhood.s01.re = segment_neighborhood.s01.re.reversed()
            segment_neighborhood.s01.im = segment_neighborhood.s01.im.reversed()
            segment_neighborhood.s11.re = segment_neighborhood.s11.re.reversed()
            segment_neighborhood.s11.im = segment_neighborhood.s11.im.reversed()
            swap(segment_neighborhood.s00, segment_neighborhood.s11)
            swap(segment_neighborhood.s10, segment_neighborhood.s01)

            # and the x-in-range mask too
            segment_neighborhood.in_range_x_mask = segment_neighborhood.in_range_x_mask.shift_right[1]()
            segment_neighborhood.in_range_x_mask[0] = True
            segment_neighborhood.in_range_x_mask = segment_neighborhood.in_range_x_mask.reversed()

    fn _voxels_bounds[*, rounding: Int = 0](
        self,
        coords_proj: FFTCoords[2],
        projections: List[VolumeNeighborhoodsProjection[dtype]],
        out bounds: Tuple[Vec[3,Int],Vec[3,Int]]
    ):
        # TODO: write tests for this specifically!

        # compute the extents of the projection grid in volume space
        var f_v_minf = Vec[3,Scalar[dtype]](fill=inf[dtype]())
        var f_v_maxf = Vec[3,Scalar[dtype]](fill=-inf[dtype]())
        for proj in projections:
            var bound = OrientedBox(
                origin = Vec[2](x=0, y=0).map_scalar[dtype]().lift(z=0),
                sizes = coords_proj.sizes_fourier().map_scalar[dtype]().lift(z=0),
                orientation = proj.rot_proj_to_vol.copy()
            ).bounding_box()
            var offset = proj.proj_to_vol(Vec[2](x=0, y=coords_proj.fmin[1]()).map_scalar[dtype]())
                .round[rounding]()
            f_v_minf = f_v_minf.min(bound.origin + offset)
            f_v_maxf = f_v_maxf.max(bound.max() + offset)

        # discretize the bounds for iteration
        var f_v_mini = f_v_minf.floor().map_int()
        var f_v_maxi = f_v_maxf.ceil().map_int()

        # fold the -x halfspace over the yz plane to push out the bounds on the +x side
        f_v_maxi.x() = max(f_v_maxi.x(), -f_v_mini.x() - 1)
        f_v_mini.x() = max(f_v_mini.x(), 0)
        # need to push out x,y too, in both directions, to account for the inversion symmetry
        @parameter
        for d in range(1, 3):
            f_v_mini[d] = min(f_v_mini[d], -f_v_maxi[d] - 1)
            f_v_maxi[d] = max(f_v_maxi[d], -f_v_mini[d] - 1)

        bounds = (f_v_mini^, f_v_maxi^)

    fn scan[
        func: fn(proj_inf: Int, var f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]) capturing,
        *,
        rounding: Int = 0,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        sizes_real_proj: Vec[2,Int],
        projections: List[VolumeNeighborhoodsProjection[dtype]],
        freq_limits: FrequencyLimits[dtype] = FrequencyLimits[dtype].none()
    ):
        @parameter
        if debug:
            ref dbg = debugger()[]
            dbg.log(String("debugging scan:",
                "  proj_i=", dbg.proj_i,
                "  f_pi=", dbg.f_pi,
                "  f_vi=", dbg.f_vi
            ))

        # TEMP
        from cryoluge.time import Profiler
        var p = Profiler[enabled=False](unit='us')
        p.start('scan')

        # impose limits on the number of projections, so we can let the compiler optimize storage better
        # (ie, use stack space instead of heap space)
        # typically, a tilt series will have ~40-60 tilts,
        # so pick a multiple of 16 (max simd width) that's a bit more than that
        comptime max_num_projections = 64  # 4 groups @ simd_width=16
        comptime Projections = _Projections[max_num_projections,simd_width,rounding=rounding]

        p.start('setup')  # TEMP
        var coords_proj = FFTCoords(sizes_real_proj)
        var freq_limits_proj = freq_limits.checker(sizes_real_proj)
        var simd_projections = Projections(projections)
        p.stop('setup')  # TEMP

        # iterate over the voxel coords that cover the projection range
        p.start('v-bounds')  # TEMP
        var bounds = self._voxels_bounds[rounding=rounding](coords_proj, projections)
        ref f_vi_min = bounds[0]
        ref f_vi_max = bounds[1]
        p.stop('v-bounds')  # TEMP

        # TEMP
        ref p_compute = p.counter('compute')
        ref p_y_bounds = p.counter('y_bounds')
        ref p_advance = p.counter('advance')
        ref p_bounds = p.counter('bounds')
        ref p_x_bounds = p.counter('x_bounds')
        ref p_samples = p.counter('samples')
        ref p_scalar = p.counter('scalar')

        # allocate space for intersection information
        var intersection_groups = InlineArray[_PIntersections[dtype,simd_width], Projections.max_num_groups](fill=_PIntersections[dtype,simd_width]())
        var bound_groups = InlineArray[_PBound[2,dtype,simd_width], Projections.max_num_groups](fill=_PBound[2,dtype,simd_width]())

        for x_vi in range(f_vi_min.x(), f_vi_max.x() + 1, Self.num_neighborhoods_in_segment):
            for z_vi in range(f_vi_min.z(), f_vi_max.z() + 1):

                p_y_bounds.start()  # TEMP

                # calculate y-bounds in volume-space for all the projections
                var fy_vf_min = inf[dtype]()
                var fy_vf_max = -inf[dtype]()
                for g in range(simd_projections.num_groups):
                    ref proj_group = simd_projections.groups[g]
                    var bound_fy_vf = proj_group.bound_fy_vf(x_vi, z_vi)
                    fy_vf_min = min(fy_vf_min, bound_fy_vf[0])
                    fy_vf_max = max(fy_vf_max, bound_fy_vf[1])

                # discretize and intersect with the y-bounds from before
                var fy_vi_min = f_vi_min.y()
                var fy_vi_max = f_vi_max.y()
                if not isinf(fy_vf_min):
                    fy_vi_min = max(fy_vi_min, Int(floor(fy_vf_min)))
                if not isinf(fy_vf_max):
                    fy_vi_max = min(fy_vi_max, Int(floor(fy_vf_max)))

                p_y_bounds.stop()  # TEMP

                @parameter
                if debug:
                    ref dbg = debugger()[]
                    if dbg.is_segment_xz[1,simd_width](x_vi, z_vi):
                        dbg.log(String("bound_fy_vf():",
                            "  fy_vf=", fy_vf_min, ",", fy_vf_max,
                            "  fy_vi=", fy_vi_min, ",", fy_vi_max
                        ))

                # compute intersections for this xz-line, one before the y-min
                p_compute.start()  # TEMP
                var f_vi_pos = Vec[3](x=x_vi, y=fy_vi_min - 1, z=z_vi)
                for g in range(simd_projections.num_groups):
                    ref proj_group = simd_projections.groups[g]
                    ref intersections = intersection_groups[g]
                    intersections.compute(f_vi_pos, proj_group)
                p_compute.stop()  # TEMP

                for y_vi in range(fy_vi_min, fy_vi_max + 1):
                    f_vi_pos.y() = y_vi

                    # advance all the intersections along y
                    p_advance.start()  # TEMP
                    for g in range(simd_projections.num_groups):
                        ref proj_group = simd_projections.groups[g]
                        ref intersections = intersection_groups[g]
                        intersections.advance_y(proj_group)
                    p_advance.stop()  # TEMP

                    # compute a bound on the intersection of the segment with the z_p=0 plane
                    p_bounds.start()  # TEMP
                    for g in range(simd_projections.num_groups):
                        ref proj_group = simd_projections.groups[g]
                        ref intersections = intersection_groups[g]
                        bound_groups[g] = intersections.bound_f_pf(proj_group)
                    p_bounds.stop()  # TEMP

                    # map to both positive and negative x halfspaces
                    @parameter
                    for x_halfspace in [1, -1]:
                        var f_vi = calc_f_vi[x_halfspace](f_vi_pos)

                        @parameter
                        if debug:
                            ref dbg = debugger()[]
                            if dbg.is_segment[x_halfspace,simd_width](f_vi):
                                dbg.log(String("iterating segment:",
                                    "  f_vi_pos=", f_vi_pos,
                                    "  x_halfspace=", x_halfspace,
                                    "  f_vi=", f_vi
                                ))

                        var segment_neighborhood: Optional[_SegmentNeighborhood[dtype,simd_width]] = None

                        # for each group of projections ...
                        for g in range(simd_projections.num_groups):
                            ref proj_group = simd_projections.groups[g]
                            ref intersections = intersection_groups[g]

                            # discretize the bounds on the intersection of the segment with the z_p=0 plane
                            p_bounds.start()  # TEMP
                            var bound_pf = bound_groups[g].copy()
                            @parameter
                            if x_halfspace == -1:
                                bound_pf.f.invert()
                            var bound_pi = proj_group.bound_pi(bound_pf, coords_proj.fmin_pos(), coords_proj.fmax())
                            var bound_all_empty = bound_pi.all_empty()
                            p_bounds.stop()  # TEMP

                            @parameter
                            if debug:
                                ref dbg = debugger()[]
                                var _w = dbg.projection_offset(proj_group)
                                if _w is not None:
                                    var w = _w.value()
                                    if dbg.is_segment[x_halfspace,simd_width](f_vi):
                                        dbg.log(String(
                                            "mask=", bound_pi.mask[w],
                                            "  bound_pf=", bound_pf.f[slice=w],
                                            "  bound_i=", bound_pi.f[slice=w]
                                        ))
                                        ref proj = projections[proj_group.proj_indices[w]]
                                        dbg.log(proj_group.render_bound_geometry[x_halfspace](w, f_vi, coords_proj))
                                        _ = intersections.bound_f_pf[debug=True, debugger=debugger](proj_group)

                            if bound_all_empty:
                                continue

                            # iterate over the projection sample lines in the y-bounds
                            var sy_simd = bound_pi.f.min.y()
                            while sy_simd.le(bound_pi.f.max.y()).reduce_or():
                                # WARNING: don't continue this loop without incrementing y!!

                                # TODO: could add a freq check here too? use 0 for x

                                # do intersection tests for each y-scanline to get tighter x-bounds
                                p_x_bounds.start()  # TEMP
                                var bound_x_pf = intersections.bound_fx_pf[x_halfspace](proj_group, sy_simd)
                                var bound_x_pi = proj_group.bound_pi(
                                    bound_x_pf,
                                    coords_proj.fmin_pos().select[0](),
                                    coords_proj.fmax().select[0]()
                                )
                                var bound_x_all_empty = bound_x_pi.all_empty()
                                p_x_bounds.stop()  # TEMP

                                @parameter
                                if debug:
                                    ref dbg = debugger()[]
                                    var _w = dbg.projection_offset(proj_group)
                                    if _w is not None:
                                        var w = _w.value()
                                        if dbg.is_segment[x_halfspace,simd_width](f_vi):
                                            dbg.log(String(
                                                "y=", sy_simd[w],
                                                "  x-mask=", bound_x_pi.mask[w],
                                                "  x-bound=", bound_x_pf.f[slice=w], " ", bound_x_pi.f[slice=w]
                                            ))
                                            _ = intersections.bound_fx_pf[x_halfspace,debug=True, debugger=debugger](proj_group, sy_simd)

                                # skip empty x-bounds
                                # (happens quite a bit actually, when the 2d bound is near an x-edge of the projection grid)
                                if bound_x_all_empty:
                                    sy_simd += 1
                                    continue

                                var sx_simd = bound_x_pi.f.min.x()
                                while sx_simd.le(bound_x_pi.f.max.x()).reduce_or():
                                    # WARNING: don't continue this loop without incrementing x!!

                                    p_samples.start()  # TEMP

                                    var sf_pi = Vec[2](x=sx_simd, y=sy_simd)
                                    var sf_pf = sf_pi.map_scalar[dtype]()

                                    # check the frequency limits
                                    var in_freq = freq_limits_proj.contains(f=sf_pf)
                                    if not in_freq.reduce_or():
                                        sx_simd += 1
                                        p_samples.stop()  # TEMP
                                        continue

                                    # transform back into reference volume space
                                    var sf_vf = proj_group.proj_to_vol(sf_pf)
                                        .round[rounding]()

                                    # (double-)check if the sample point actually lies in the segment
                                    # (treat the upper boundaries as exclusive)
                                    var seg_bounds_vf = intersections.get_seg_bounds_vf[x_halfspace]()
                                    var in_bounds = sf_vf.ge_all(seg_bounds_vf[0])
                                        &  sf_vf.lt_all(seg_bounds_vf[1])
                                    # NOTE: this check is still necessary, since the bounds aren't perfectly tight

                                    # get the x-offset into the segment and the distances
                                    var x_offset = SIMDInt[simd_width](floor(sf_vf.x())) - f_vi.x()
                                    @parameter
                                    if x_halfspace == -1:
                                        x_offset = -x_offset
                                    var dists_v = sf_vf - sf_vf.floor()

                                    p_samples.stop()  # TEMP

                                    @parameter
                                    if debug:
                                        ref dbg = debugger()[]
                                        var _w = dbg.projection_offset(proj_group)
                                        if _w is not None:
                                            var w = _w.value()
                                            if dbg.is_segment[x_halfspace,simd_width](f_vi):
                                                dbg.log(String(
                                                    "sf_pi=", sf_pi[slice=w],
                                                    "  sv_vf=", sf_vf[slice=w],
                                                    "  min=", seg_bounds_vf[0][slice=w],
                                                    "  max=", seg_bounds_vf[1][slice=w],
                                                    "  in_bounds=", in_bounds[w],
                                                    "  x_offset=", x_offset[w],
                                                    "  dists_v=", dists_v[slice=w]
                                                ))
                                            if dbg.is_sample( sf_pi[slice=w].map_int()):
                                                dbg.log(String("sf_pi found in f_vi=", f_vi))

                                    # TODO: we're dropping into scalar mode after this
                                    #       what can we still vectorize?

                                    p_scalar.start()  # TEMP

                                    # for each projection in the group ...
                                    for w in range(proj_group.num_projections):
                                        ref proj = projections[proj_group.proj_indices[w]]

                                        # apply the sample filters
                                        if not in_bounds[w]:
                                            continue
                                        if not in_freq[w]:
                                            continue

                                        var sf_pi = sf_pi[slice=w].map_int()
                                        var sf_vf = sf_vf[slice=w]
                                        var x_offset = Int(x_offset[w])
                                        var dists_v = dists_v[slice=w]

                                        # finally, interpolate the reference volume
                                        if segment_neighborhood is None:
                                            segment_neighborhood = self._segment_neighborhood[x_halfspace](f_vi_pos)

                                        var voxel_neighborhood = segment_neighborhood.value()
                                            .voxel_neighborhood[x_halfspace, Self.out_of_range](x_offset)
                                        var sv = interpolate(dists_v, voxel_neighborhood)

                                        @parameter
                                        if debug:
                                            ref dbg = debugger()[]
                                            if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                                dbg.log(String("segment_neighborhood=", _render_neighborhood(segment_neighborhood.value())))
                                                dbg.log(String(
                                                    "voxel_neighborhood=", _render_neighborhood(voxel_neighborhood),
                                                    "  sv=", sv
                                                ))

                                        func(proj.id, sf_pi^, sf_vf^, sv)

                                    # end of projections loop

                                    p_scalar.stop()  # TEMP

                                    sx_simd += 1
                                # end of x sample loop

                                sy_simd += 1
                            # end of y scanlines loop
                        # end of projection group loop
                    # end of halfspace loop
                # end of voxel loop

        # TEMP
        p.stop('scan')
        p.print()


fn _num_neighborhoods_in_segment[simd_width: Int]() -> Int:
    return simd_width - 1
    # one less neighborhood, due to needing two x voxels per neighborhood


struct VolumeNeighborhoodsProjection[dtype: DType](
    Copyable,
    Movable
):
    var id: Int
    var rot_proj_to_vol: Matrix[3,3,dtype]

    fn __init__(
        out self,
        id: Int,
        rot_proj_to_vol: Matrix[3,3,dtype]
    ):
        self.id = id
        self.rot_proj_to_vol = rot_proj_to_vol.copy()

    @always_inline
    fn proj_to_vol(
        self,
        v: Vec[3,Scalar[dtype]],
        out result: Vec[3,Scalar[dtype]]
    ):
        result = self.rot_proj_to_vol*v

    @always_inline
    fn proj_to_vol(
        self,
        v: Vec[2,Scalar[dtype]],
        out result: Vec[3,Scalar[dtype]]
    ):
        result = self.proj_to_vol(v.lift(z=0))

    @always_inline
    fn vol_to_proj[simd_width: Int](
        self,
        v: Vec[3,SIMD[dtype,simd_width]],
        out result: Vec[3,SIMD[dtype,simd_width]]
    ):
        result = self.rot_proj_to_vol.mul_transpose(v)
    

comptime _VoxelNeighborhood[dtype: DType] = ComplexSIMD[dtype,8]


struct _ProjectionGroup[dtype: DType, simd_width: Int, *, rounding: Int = 0](
    Copyable,
    Movable
):
    var num_projections: Int
    var proj_indices: SIMDInt[simd_width]
    var proj_mask: SIMDBool[simd_width]
    var rot_proj_to_vol: Matrix[3,3,dtype,simd_width]
    var segment_extents_neg: Vec[3,SIMD[dtype,simd_width]]
    var segment_extents_pos: Vec[3,SIMD[dtype,simd_width]]
    var planes_xy: Self.Planes[Self.PX,Self.PY,Self.PZ]
    var planes_yz: Self.Planes[Self.PY,Self.PZ,Self.PX]
    var planes_zx: Self.Planes[Self.PZ,Self.PX,Self.PY]

    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()
    comptime segment_sizes = Vec[3,Int](x=Self.num_neighborhoods_in_segment, y=1, z=1)
    comptime PX = _Plane.x(Self.segment_sizes.x())
    comptime PY = _Plane.y()
    comptime PZ = _Plane.z()
    comptime Planes = _Planes[dtype,simd_width,_]

    fn __init__(out self):
        comptime zero_i = SIMDInt[simd_width](0)
        comptime zero_f = SIMD[dtype,simd_width](0)
        self.num_projections = 0
        self.proj_indices = zero_i
        self.proj_mask = SIMDBool[simd_width](fill=False)
        self.rot_proj_to_vol = Matrix[3,3,dtype,simd_width](fill=0)
        self.segment_extents_neg = Vec[3](fill=zero_f)
        self.segment_extents_pos = Vec[3](fill=zero_f)
        self.planes_xy = Self.Planes[Self.PX,Self.PY,Self.PZ]()
        self.planes_yz = Self.Planes[Self.PY,Self.PZ,Self.PX]()
        self.planes_zx = Self.Planes[Self.PZ,Self.PX,Self.PY]()

    @always_inline
    fn vol_to_proj(
        self,
        f_vf: Vec[3,Scalar[dtype]],
        out f_pf: Vec[3,SIMD[dtype,simd_width]]
    ):
        f_pf = self.vol_to_proj(f_vf.splat[simd_width]())

    @always_inline
    fn vol_to_proj(
        self,
        f_vf: Vec[3,SIMD[dtype,simd_width]],
        out f_pf: Vec[3,SIMD[dtype,simd_width]]
    ):
        f_pf = self.rot_proj_to_vol.mul_transpose(f_vf)

    @always_inline
    fn proj_to_vol(
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        out f_vf: Vec[3,SIMD[dtype,simd_width]]
    ):
        f_vf = self.rot_proj_to_vol*f_pf

    @always_inline
    fn proj_to_vol(
        self,
        f_pf: Vec[2,SIMD[dtype,simd_width]],
        out f_vf: Vec[3,SIMD[dtype,simd_width]]
    ):
        f_vf = self.proj_to_vol(f_pf.lift(z=0))

    @always_inline
    fn normal_p[d: Int](
        self,
        out normal_p: Vec[3,SIMD[dtype,simd_width]]
    ):
        normal_p = self.rot_proj_to_vol.vec(row=d)

    @always_inline
    fn normal_v[d: Int](
        self,
        out normal_v: Vec[3,SIMD[dtype,simd_width]]
    ):
        normal_v = self.rot_proj_to_vol.vec(col=d)

    fn bound_fy_vf(
        self,
        x_vi: Int,
        z_vi: Int,
        out bound_fy_vf: Tuple[Scalar[dtype],Scalar[dtype]]
    ):
        var fy_vf_min = inf[dtype]()
        var fy_vf_max = -inf[dtype]()

        # calculate the segment bounds at this x,z position
        var seg_min_vf = Vec[3](x=x_vi, y=0, z=z_vi)
        var seg_max_vf = seg_min_vf + materialize[Self.segment_sizes]()
        var seg_bounds_vf = (
            seg_min_vf.map_scalar[dtype]().splat[simd_width](),
            seg_max_vf.map_scalar[dtype]().splat[simd_width]()
        )

        # and the fy_vf terms (d3=y in this case)
        var fy_vf_terms = self.planes_zx.d3_vf_terms(seg_bounds_vf)

        @parameter
        for uv in _UV.all:

            # get the y-coordinate in volume space of the z,x planes intersection point with z_p=0
            var fy_vf = fy_vf_terms[uv.u][0] + fy_vf_terms[uv.v][1]

            var any_bad = ((isinf(fy_vf) | isnan(fy_vf)) & self.proj_mask).reduce_or()
            if any_bad:
                # points at infinity here means the range is infinite =(
                return (-inf[dtype](), inf[dtype]())

            # replace any out-of-projections values with NaN, so they don't affect the min,max
            fy_vf = self.proj_mask.select(fy_vf, SIMD[dtype,simd_width](nan[dtype]()))

            # update the bounds
            fy_vf_min = min(fy_vf_min, fy_vf.reduce_min())
            fy_vf_max = max(fy_vf_max, fy_vf.reduce_max())

        bound_fy_vf = (fy_vf_min, fy_vf_max)

    fn bound_pi[dim: Int, bound_simd_width: Int](
        self,
        bound_pf: _PBound[dim,dtype,bound_simd_width],
        min_pi: Vec[dim,Int],
        max_pi: Vec[dim,Int],
        out bound_pi: _PBound[dim,DType.int,bound_simd_width]
    ):
        bound_pi = _PBound[dim,DType.int,bound_simd_width]()

        # the mask needs no changes
        bound_pi.mask = bound_pf.mask

        ref bf = bound_pf.f
        ref bi = bound_pi.f

        # apply rounding before discretizing, if needed
        var bf_min = bf.min.round[rounding]()
        var bf_max = bf.max.round[rounding]()

        # discretize the bound, paying attention to the inclusivity of each boundary
        @parameter
        for d in range(dim):
            bi.min[d] = bf.min_inclusive[d].select(
                true_case = SIMDInt[bound_simd_width]( ceil(bf_min[d]) ),
                false_case = SIMDInt[bound_simd_width]( floor(bf_min[d] + 1) )
            )
            bi.max[d] = bf.max_inclusive[d].select(
                true_case = SIMDInt[bound_simd_width]( floor(bf_max[d]) ),
                false_case = SIMDInt[bound_simd_width]( ceil(bf_max[d] - 1) )
            )

        # intersect with the projection bounds
        var f_min_p = min_pi.map_scalar[DType.int]().splat[bound_simd_width]()
        var f_max_p = max_pi.map_scalar[DType.int]().splat[bound_simd_width]()
        bi.min = bi.min.max(f_min_p)
        bi.max = bi.max.min(f_max_p)

        # the above logic creates fully-inclusive integer bounds
        bi.min_inclusive = Vec[dim,SIMDBool[bound_simd_width]](fill=SIMDBool[bound_simd_width](fill=True))
        bi.max_inclusive = Vec[dim,SIMDBool[bound_simd_width]](fill=SIMDBool[bound_simd_width](fill=True))

    # for debugging
    fn render_bound_geometry[x_halfspace: Int](
        self,
        w: Int,
        f_vi: Vec[3,Int],
        coords_proj: FFTCoords[2],
        out str: String
    ):
        var f_vi_corner = calc_f_vi_corner[x_halfspace,simd_width](f_vi)
        var f_vf_corner = f_vi_corner.map_scalar[dtype]().splat[simd_width]()

        # compute all the intersection points
        var intersections = _PIntersections[dtype,simd_width]()
        intersections.compute(f_vi_corner, self)

        # classify all the intersection points
        var inside_points = List[Vec[2,Scalar[dtype]]]()
        var outside_points = List[Vec[2,Scalar[dtype]]]()

        @parameter
        fn classify_intersection[uv: _UV, p1: _Plane, p2: _Plane, p3: _Plane](
            planes: Self.Planes[p1,p2,p3]
        ):
            var i = intersections.intersect[uv](self, planes)
            if i.in_range[w]:
                inside_points.append(i.point[slice=w])
            else:
                outside_points.append(i.point[slice=w])

        # collect all 12 intersection points
        @parameter
        for uv in _UV.all:
            classify_intersection[uv](self.planes_xy)
            classify_intersection[uv](self.planes_yz)
            classify_intersection[uv](self.planes_zx)

        # compute the p-bounds
        var bound_pf = intersections.bound_f_pf(self)
        @parameter
        if x_halfspace == -1:
            bound_pf.f.invert()
        var bound_pi = self.bound_pi(bound_pf, coords_proj.fmin_pos(), coords_proj.fmax())

        # compute the x p-bounds
        var x_in = List[Vec[2,Scalar[dtype]]]()
        var x_out = List[Vec[2,Scalar[dtype]]]()

        @parameter
        fn classify_x_intersection[i: Int, p1: _Plane, p2: _Plane, p3: _Plane](
            planes: Self.Planes[p1,p2,p3],
            fy_pf: Scalar[dtype]
        ):
            var dot_bounds_x = intersections.dot_bounds_x[x_halfspace](self, fy_pf)
            var (in_range, _inclusive, fx_pf) = intersections.intersect_x[x_halfspace,i](self, planes, fy_pf, dot_bounds_x)
            var p_pf = Vec[2](x=fx_pf[w], y=fy_pf)
            if in_range[w]:
                x_in.append(p_pf^)
            else:
                x_out.append(p_pf^)

        for fy_pi in range(Int(bound_pi.f.min[1][w]), Int(bound_pi.f.max[1][w]) + 1):
            var fy_pf = Scalar[dtype](fy_pi)

            ref pxy = self.planes_xy
            ref pyz = self.planes_yz
            ref pzx = self.planes_zx

            @parameter
            for i in [0,1]:
                classify_x_intersection[i](pxy, fy_pf)
                classify_x_intersection[i](pyz, fy_pf)
                classify_x_intersection[i](pzx, fy_pf)

        @parameter
        fn display_intersections(
            pts: List[Vec[2,Scalar[dtype]]],
            out s: String
        ):
            s = ""
            for p in pts:
                s += "\n\tpt"
                s += String(p)
                s += ","

        str = String("Plane bound geometry:",
            "\n# f_vi=", f_vi,
            "\n# f_vi_corner=", f_vi_corner,
            "\ngrid_p=[", coords_proj.fmin_pos(), ",", coords_proj.fmax(), "]",
            "\nf_pf=pt", self.vol_to_proj(f_vi.map_scalar[dtype]())[slice=w],
            "\nf_pf_corner=pt", self.vol_to_proj(f_vf_corner)[slice=w],
            "\naxes=[",
                "\n\tpt", self.normal_p[0]()[slice=w], ","
                "\n\tpt", self.normal_p[1]()[slice=w], ","
                "\n\tpt", self.normal_p[2]()[slice=w],
            "\n]",
            "\nx_halfspace=", x_halfspace,
            "\nx_len=", _num_neighborhoods_in_segment[simd_width](),
            "\nintersections_in=[",
                display_intersections(inside_points),
            "\n]",
            "\nintersections_out=[",
                display_intersections(outside_points),
            "\n]",
            "\nbound_pf=[",
                "pt", bound_pf.f.min[slice=w],
                ", pt", bound_pf.f.max[slice=w],
            "]",
            "\nbound_pi=[",
                "pt", bound_pi.f.min[slice=w],
                ", pt", bound_pi.f.max[slice=w],
            "]",
            "\nintersections_x_in=[",
                display_intersections(x_in),
            "\n]",
            "\nintersections_x_out=[",
                display_intersections(x_out),
            "\n]",
            "\npuv_xy=[",
                "pt", intersections.points_pf[self.planes_xy][slice=w],
                ", pt", self.planes_xy.u[slice=w],
                ", pt", self.planes_xy.v[slice=w],
            "]",
            "\npuv_yz=[",
                "pt", intersections.points_pf[self.planes_yz][slice=w],
                ", pt", self.planes_yz.u[slice=w],
                ", pt", self.planes_yz.v[slice=w],
            "]",
            "\npuv_zx=[",
                "pt", intersections.points_pf[self.planes_zx][slice=w],
                ", pt", self.planes_zx.u[slice=w],
                ", pt", self.planes_zx.v[slice=w],
            "]"
        )

    fn render_bound_geometry(
        self,
        w: Int,
        f_vi: Vec[3,Int],
        coords_proj: FFTCoords[2],
        out str: String,
        *,
        x_halfspace: Int
    ):
        if x_halfspace == 1:
            str = self.render_bound_geometry[1](w, f_vi, coords_proj)
        else:
            str = self.render_bound_geometry[-1](w, f_vi, coords_proj)


struct _Projections[max_num_projections: Int, simd_width: Int, dtype: DType, *, rounding: Int = 0](
    Copyable,
    Movable
):
    var num_groups: Int
    var groups: InlineArray[Self.Group,Self.max_num_groups]

    comptime max_num_groups = ceildiv(max_num_projections, simd_width)
    comptime Group = _ProjectionGroup[dtype,simd_width,rounding=rounding]

    fn __init__(out self, projections: List[VolumeNeighborhoodsProjection[dtype]]):

        # check the sizes
        self.num_groups = ceildiv(len(projections), simd_width)
        if self.num_groups > Self.max_num_groups:
            abort(String("Too many projections: ", len(projections), ", max allowed is ", max_num_projections))

        # allocate all the groups
        self.groups = InlineArray[Self.Group,Self.max_num_groups](fill=Self.Group())

        # populate the groups with each projection
        for p in range(len(projections)):
            ref proj = projections[p]
            var g = p // simd_width
            ref group = self.groups[g]
            var i = p % simd_width

            group.num_projections += 1
            group.proj_indices[i] = p
            group.proj_mask[i] = True

            # pack the rotation matrices
            group.rot_proj_to_vol[slice=i] = proj.rot_proj_to_vol

            comptime n_i = _num_neighborhoods_in_segment[simd_width]()
            comptime n_f = Scalar[dtype](n_i)

            # compute the bounding volume extents of the unit voxel,
            # in projection space, relative to the voxel origin
            var voxel_bound = OrientedBox(
                origin = Vec[3](fill=Scalar[dtype](0)),
                sizes = Vec[3](fill=Scalar[dtype](1)),
                orientation = proj.rot_proj_to_vol.transposed()
            ).bounding_box()
            var voxel_extents_neg = voxel_bound.origin.copy()
            var voxel_extents_pos = voxel_bound.max()

            # pack the segment bounding box extents
            var seg_vec = proj.rot_proj_to_vol.vec(row=0)*(n_f - 1)
            group.segment_extents_neg[slice=i] = voxel_extents_neg.min(voxel_extents_neg + seg_vec)
            group.segment_extents_pos[slice=i] = voxel_extents_pos.max(voxel_extents_pos + seg_vec)

        # init the plane pairs
        for i in range(len(self.groups)):
            ref g = self.groups[i]
            var pxy = _Planes[dtype,simd_width,Self.Group.PX,Self.Group.PY,Self.Group.PZ]()
            var pyz = _Planes[dtype,simd_width,Self.Group.PY,Self.Group.PZ,Self.Group.PX]()
            var pzx = _Planes[dtype,simd_width,Self.Group.PZ,Self.Group.PX,Self.Group.PY]()
            pxy.init(g)
            pyz.init(g)
            pzx.init(g)
            # NOTE: other planes is the one where p1 is in the p2 position
            pxy.init_more(pzx)
            pyz.init_more(pxy)
            pzx.init_more(pyz)
            g.planes_xy = pxy^
            g.planes_yz = pyz^
            g.planes_zx = pzx^


@fieldwise_init
struct _UV(
    ImplicitlyCopyable,
    Movable,
    Writable,
    Stringable
):
    var u: Int
    var v: Int

    comptime all = [_UV(0,0), _UV(0,1), _UV(1,0), _UV(1,1)]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.u, ",", self.v)

    fn __str__(self) -> String:
        return String.write(self)


struct _UVMap[T: AnyType & Copyable & Movable](
    Copyable,
    Movable
):
    var _values: InlineArray[T,Self.num_values]

    comptime num_values = 4

    fn __init__(out self, *, fill: T):
        self._values = InlineArray[T,Self.num_values](fill=fill)

    # TODO: upgrade these to __getitem_parm__ in the new versions of Mojo
    @always_inline
    fn at[u: Int, v: Int](ref self) -> ref [self._values] T:

        # just in case ...
        constrained[
            u in [0,1] and v in [0,1],
            String("invalid u,v: ", u, ",", v)
        ]()

        return self._values[(u<<1) | v]

    @always_inline
    fn at[uv: _UV](ref self) -> ref [self._values] T:
        return self.at[uv.u, uv.v]()


@fieldwise_init
struct _Plane(
    Copyable,
    Movable
):
    var name: StaticString
    var d: Int
    var len: Int

    @staticmethod
    fn x(out self: Self, len: Int=1):
        self = Self("x", 0, len)

    @staticmethod
    fn y(out self: Self, len: Int=1):
        self = Self("y", 1, len)

    @staticmethod
    fn z(out self: Self, len: Int=1):
        self = Self("z", 2, len)

    fn normal_i(self, out n: Vec[3,Int]):
        n = Vec[3](fill=0)
        n[self.d] = 1

    fn normal_f[dtype: DType](self, out n: Vec[3,Scalar[dtype]]):
        n = self.normal_i().map_scalar[dtype]()


struct _Planes[dtype: DType, simd_width: Int, p1: _Plane, p2: _Plane, p3: _Plane](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    # TODO: rename these to something more meaningful!
    var f: Vec[2,SIMD[dtype,simd_width]]
    var s1: Vec[3,SIMD[dtype,simd_width]]
    var s2: Vec[3,SIMD[dtype,simd_width]]
    var u: Vec[2,SIMD[dtype,simd_width]]
    var v: Vec[2,SIMD[dtype,simd_width]]
    var uv_selector: Vec[2,SIMDBool[simd_width]]
    var v_xoy_pf: SIMD[dtype,simd_width]
    var u_xmy_pf: SIMD[dtype,simd_width]

    comptime Group = _ProjectionGroup[dtype,simd_width,rounding=_]
    comptime Planes = _Planes[dtype,simd_width]

    fn __init__(out self):
        comptime zero = SIMD[dtype,simd_width](0)
        self.f = Vec[2](fill=zero)
        self.s1 = Vec[3](fill=zero)
        self.s2 = Vec[3](fill=zero)
        self.u = Vec[2](fill=zero)
        self.v = Vec[2](fill=zero)
        self.uv_selector = Vec[2](fill=SIMDBool[simd_width](fill=False))
        self.v_xoy_pf = zero
        self.u_xmy_pf = zero

    fn init(mut self, group: Self.Group):

        # intersection forumla for two axis-aligned volume-space planes with z_p=0,
        # but only the third coordinate
        # ie, at a point p_v, the third coord is p_v.f
        var nz_vf = group.normal_v[2]()
        self.f = -self.project12(nz_vf)/nz_vf[p3.d]

        # get the plane normals and mix them
        var n0 = group.normal_p[p1.d]()
        var n1 = group.normal_p[p2.d]()
        var n00 = n0.inner_product(n0)
        var n01 = n0.inner_product(n1)
        var n11 = n1.inner_product(n1)

        # intersection formula for two planes in projection-space at z_p=0
        # ie, at a point p_p, the intersection is <p_p.t1, p_p.t2>
        var d = n1[0]*n0[1] - n0[0]*n1[1]
        var u1 = Vec[2](x=-n1[1], y=n0[1])/d
        var u2 = Vec[2](x=n1[0], y=-n0[0])/d
        var t1 = u1.three_inner_products(
            Vec[2](x=n0[0], y=n1[0]),
            Vec[2](x=n0[1], y=n1[1]),
            Vec[2](x=n0[2], y=n1[2])
        )
        var t2 = u2.three_inner_products(
            Vec[2](x=n0[0], y=n1[0]),
            Vec[2](x=n0[1], y=n1[1]),
            Vec[2](x=n0[2], y=n1[2])
        )

        # rotate t1,t2 into volume space so we can apply them to volume-space points directly
        # ie, at a point p_v, the intersection is <p_v.s1, p_v.s2>
        self.s1 = group.proj_to_vol(t1)
        self.s2 = group.proj_to_vol(t2)

        # direction vectors for the volume-space unit axes in the z_p=0 plane
        # ie, for a delta vector <dx_v,dy_v> in volume-space,
        # the intersection point moves by <dx_v*u, dy_v*v> in projection-space
        # NOTE: u is orthogonal to the p2 normal,
        #       and v is orthogonal to the p1 normal
        #       meaning, u points to the far parallel parner of plane 1
        #       and v points to the far parallel partner of plane 2
        self.u = Vec[2](x=n00, y=n01).two_inner_products(u1, u2)*p1.len
        self.v = Vec[2](x=n01, y=n11).two_inner_products(u1, u2)*p2.len
        # TODO: u,v are redundant, ie two copies of each (except for an inversion)
        #       could put in single-plane storage? and negate as needed?

    fn init_more(mut self, other_planes: Self.Planes[_,_,_]):

        ref v1 = self.v
        ref v2 = other_planes.u

        # Tragically, the intersection geometry can have points at essentially infinity,
        # so we have to be careful about using the u,v vectors.
        # So, to be numerically stable, pick the u,v vectors that are well-behaved,
        # ie, actually defined, and not suuuper large (near infinity)
        var use_1 = ~v1.has_nan_simd() & (v2.has_nan_simd() | v1.len2().lt(v2.len2()))
        self.uv_selector = Vec[2](fill=use_1)

        # pick the u,v vectors for the subsequent calculations
        ref u1 = self.u
        ref u2 = other_planes.v
        var u = self.uv_selector.select(u1, u2)
        var v = self.uv_selector.select(v1, v2)

        # compute factors for y-scanline intersection
        self.v_xoy_pf = v.x()/v.y()
        self.u_xmy_pf = u.x() - u.y()*self.v_xoy_pf

    @always_inline
    fn project12[vec_simd_width: Int](
        self,
        v: Vec[3,SIMD[dtype,vec_simd_width]],
        out result: Vec[2,SIMD[dtype,vec_simd_width]]
    ):
        result = Vec[2](x=v[p1.d], y=v[p2.d])

    @always_inline
    fn project23[vec_simd_width: Int](
        self,
        v: Vec[3,SIMD[dtype,vec_simd_width]],
        out result: Vec[2,SIMD[dtype,vec_simd_width]]
    ):
        result = Vec[2](x=v[p2.d], y=v[p3.d])

    @always_inline
    fn d3_vf_terms(
        self,
        seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]],
        out d3_vf_terms: Tuple[Vec[2,SIMD[dtype,simd_width]],Vec[2,SIMD[dtype,simd_width]]]
    ):
        d3_vf_terms = (
            self.project12(seg_bounds_vf[0])*self.f,
            self.project12(seg_bounds_vf[1])*self.f
        )

    @always_inline
    fn intersect_min_p(
        self,
        seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]],
        out p: Vec[2,SIMD[dtype,simd_width]]
    ):
        p = seg_bounds_vf[0].two_inner_products(self.s1, self.s2)

    @always_inline
    fn intersect_p[uv: _UV](
        self,
        p_pf: Vec[2,SIMD[dtype,simd_width]],
        out i_pf: Vec[2,SIMD[dtype,simd_width]]
    ):
        # compute the intersection point, in projection-space,
        # by just translating the existing intersection point along the u,v directions
        i_pf = p_pf + self.u*uv.u + self.v*uv.v

    @always_inline
    fn sy(self, out sy: Vec[2,SIMD[dtype,simd_width]]):
        sy = Vec[2](x=self.s1.y(), y=self.s2.y())

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(p1.name, p2.name)

    fn __str__(self) -> String:
        return String.write(self)


struct _PlaneMap[dim: Int, T: AnyType & Copyable & Movable](
    Copyable,
    Movable
):
    var _values: InlineArray[T,dim]

    fn __init__(out self, *, fill: T):
        self._values = InlineArray[T,dim](fill=fill)

    # TODO: upgrade these to __getitem_parm__ in the new versions of Mojo
    @always_inline
    fn at[pi: Int](ref self) -> ref [self._values] T:

        # just in case ...
        constrained[
            pi in [0,1,2],
            String("invalid plane index: ", pi)
        ]()

        return self._values[pi]

    @always_inline
    fn at[p: _Plane](ref self) -> ref [self._values] T:
        return self.at[p.d]()

    @always_inline
    fn __getitem__[p3: _Plane](ref self, planes: _Planes[_,_,_,_,p3]) -> ref [self._values] T:
        return self.at[p3]()


struct _PIntersections[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    # TODO: seg_bounds isn't per projection group, right?
    #       could move outside
    var seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]]
    var points_pf: _PlaneMap[3,Vec[2,SIMD[dtype,simd_width]]]
    var q_xmy_pf: _PlaneMap[3,SIMD[dtype,simd_width]]

    comptime Group = _ProjectionGroup[dtype,simd_width,rounding=_]
    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()
    comptime segment_sizes = Vec[3,Int](x=Self.num_neighborhoods_in_segment, y=1, z=1)
    comptime Planes = _Planes[dtype,simd_width,_,_,_]

    fn __init__(out self):

        comptime zero = SIMD[dtype,simd_width](0)
        comptime zero2 = Vec[2,SIMD[dtype,simd_width]](fill=0)
        comptime zero3 = Vec[3,SIMD[dtype,simd_width]](fill=0)

        self.seg_bounds_vf = materialize[(zero3, zero3)]()
        self.points_pf = _PlaneMap[3](fill=materialize[zero2]())
        self.q_xmy_pf = _PlaneMap[3](fill=zero)

    fn compute(
        mut self,
        f_vi_corner: Vec[3,Int],
        group: Self.Group
    ):
        # get the segment bounds in volume-space
        var min_vf = f_vi_corner.map_scalar[dtype]().splat[simd_width]()
        var max_vf = (min_vf + materialize[Self.segment_sizes.map_scalar[dtype]().splat[simd_width]()]())
        self.seg_bounds_vf = (min_vf^, max_vf^)

        ref pxy = group.planes_xy
        ref pyz = group.planes_yz
        ref pzx = group.planes_zx

        # compute the intersection points themselves
        self.points_pf[pxy] = pxy.intersect_min_p(self.seg_bounds_vf)
        self.points_pf[pyz] = pyz.intersect_min_p(self.seg_bounds_vf)
        self.points_pf[pzx] = pzx.intersect_min_p(self.seg_bounds_vf)

        # compute y-scanline intersection factors, in projection-space
        # NOTE: other planes is the one where p1 is in the p2 position
        var q = pxy.uv_selector.select(self.points_pf[pxy], self.points_pf[pzx])
        self.q_xmy_pf[pxy] = q.x() - q.y()*pxy.v_xoy_pf
        q = pyz.uv_selector.select(self.points_pf[pyz], self.points_pf[pxy])
        self.q_xmy_pf[pyz] = q.x() - q.y()*pyz.v_xoy_pf
        q = pzx.uv_selector.select(self.points_pf[pzx], self.points_pf[pyz])
        self.q_xmy_pf[pzx] = q.x() - q.y()*pzx.v_xoy_pf

    fn advance_y(
        mut self,
        group: Self.Group
    ):
        # update the segment bounds
        self.seg_bounds_vf[0].y() += 1
        self.seg_bounds_vf[1].y() += 1

        ref pxy = group.planes_xy
        ref pyz = group.planes_yz
        ref pzx = group.planes_zx

        # advance only the intersection points that are affected by y_v
        self.points_pf[pxy] += pxy.sy()
        self.points_pf[pyz] += pyz.sy()

        # re-compute y-scanline intersection factors, in projection space
        # NOTE: can't do delta updates here, since q could be either of two intersection points,
        #       one of which may not have advanced
        var q = pxy.uv_selector.select(self.points_pf[pxy], self.points_pf[pzx])
        self.q_xmy_pf[pxy] = q.x() - q.y()*pxy.v_xoy_pf
        q = pyz.uv_selector.select(self.points_pf[pyz], self.points_pf[pxy])
        self.q_xmy_pf[pyz] = q.x() - q.y()*pyz.v_xoy_pf

    @always_inline
    fn get_seg_bounds_vf[x_halfspace: Int](
        self,
        out seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]]
    ):
        seg_bounds_vf = self.seg_bounds_vf.copy()
        @parameter
        if x_halfspace == -1:
            var swap = -seg_bounds_vf[0]
            seg_bounds_vf[0] = -seg_bounds_vf[1]
            seg_bounds_vf[1] = swap^

    @always_inline
    fn in_range[
        p3: _Plane, rounding: Int = 0,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group[rounding=rounding],
        planes: Self.Planes[_,_,p3],
        f_pf: Vec[2,SIMD[dtype,simd_width]],
        out in_range: SIMDBool[simd_width]
    ):
        # determine if the point lies inside the segment
        # by checking dot-products with the third plane normal
        var f_dot_n = f_pf.inner_product(group.normal_p[p3.d]().project[2]())
        var f_dot_n_rounded = round[rounding](f_dot_n)
        in_range = f_dot_n_rounded.ge(self.seg_bounds_vf[0][p3.d])
            & f_dot_n_rounded.le(self.seg_bounds_vf[1][p3.d])

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("in_range()",
                    "  planes=", planes,
                    "  f_pf=", f_pf[slice=w],
                    "  dot=", f_dot_n[w],
                    "  bounds=[", self.seg_bounds_vf[0][p3.d][w], ",", self.seg_bounds_vf[1][p3.d][w], "]",
                    "  in_range=", in_range[w]
                ))

    @always_inline
    fn intersect[
        uv: _UV, p3: _Plane, rounding: Int = 0,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group[rounding=rounding],
        planes: Self.Planes[_,_,p3],
        out result: _Intersection[2,dtype,simd_width]
    ):
        var f_pf = self.points_pf[planes] + planes.u*uv.u + planes.v*uv.v
        var in_range = self.in_range[debug=debug,debugger=debugger](group, planes, f_pf)
        result = _Intersection(in_range, f_pf^)

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("intersect()",
                    "  uv=", uv,
                    "  planes=", planes,
                    "  f_pf=", result.point[slice=w],
                    "  in_range=", result.in_range[w]
                ))

    fn bound_f_pf[
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group,
        out bound_pf: _PBound[2,dtype,simd_width]
    ):
        ref pxy = group.planes_xy
        ref pyz = group.planes_yz
        ref pzx = group.planes_zx

        bound_pf = _PBound[2,dtype,simd_width]()
        @parameter
        for uv in _UV.all:
            bound_pf.update(self.intersect[uv,debug=debug,debugger=debugger](group, pxy))
            bound_pf.update(self.intersect[uv,debug=debug,debugger=debugger](group, pyz))
            bound_pf.update(self.intersect[uv,debug=debug,debugger=debugger](group, pzx))

        # start with the bounds being inclusive by default
        bound_pf.f.set_inclusive(True)

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("bound_f_pf()",
                    "  bound_pf=", bound_pf.f[slice=w],
                ))

    @always_inline
    fn ranges_x[
        p2: _Plane, p3: _Plane, rounding: Int = 0,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group[rounding=rounding],
        planes: Self.Planes[_,p2,p3],
        x_pf: SIMD[dtype,simd_width],
        dot_bounds: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]],
        out result: Tuple[SIMDBool[simd_width],SIMDBool[simd_width]]
    ):
        var dot_min = planes.project23(dot_bounds[0])
        var dot_max = planes.project23(dot_bounds[1])

        # determine if the point lies inside the segment
        # by checking dot-products with the second and third plane normals
        var dot_x = Vec[2](
            x=group.normal_p[p2.d]().x(),
            y=group.normal_p[p3.d]().x()
        )*x_pf
        var dot_x_rounded = dot_x.round[rounding]()
        var in_range = dot_x_rounded.ge_all(dot_min)
            & dot_x_rounded.le_all(dot_max)

        # a point is exclusive iff it lies exactly on the upper bound,
        # so the point is inclusive if it doesn't
        var inclusive = dot_x_rounded.ne_all_simd(dot_max)

        result = (in_range, inclusive)

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("ranges_x()",
                    "  planes=", p2.name, p3.name,
                    "  x_pf=", x_pf[w],
                    "  dot_x=", dot_x[slice=w],
                    "  range=[", dot_bounds[0][slice=w], ",", dot_bounds[1][slice=w], "]",
                    "  in_range=", in_range[w]
                ))

    @always_inline
    fn intersect_x[
        x_halfspace: Int,
        i: Int, p1: _Plane, p2: _Plane, p3: _Plane, rounding: Int,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group[rounding=rounding],
        planes: Self.Planes[p1,p2,p3],
        y_pf: SIMD[dtype,simd_width],
        dot_bounds: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]],
        out result: Tuple[SIMDBool[simd_width],SIMDBool[simd_width],SIMD[dtype,simd_width]]
    ):
        var y_pf_hs = y_pf.copy()
        @parameter
        if x_halfspace == -1:
            y_pf_hs = -y_pf_hs

        # solve for x at the given y, in projection space
        var x_pf = self.q_xmy_pf[planes] + y_pf_hs*planes.v_xoy_pf + i*planes.u_xmy_pf
        @parameter
        if x_halfspace == -1:
            x_pf *= -1

        (in_range, inclusive) = self.ranges_x[debug=debug,debugger=debugger](group, planes, x_pf, dot_bounds)

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("intersect_x()",
                    "  x_halfspace=", x_halfspace,
                    "  planes=", p1.name, p2.name, p3.name,
                    "  i=", i,
                    "  y_pf=", y_pf[w],
                    "  y_pf_hs=", y_pf_hs[w],
                    "  x_pf=", x_pf[w],
                    "  in_range=", in_range[w],
                    "  inclusive=", inclusive[w]
                ))

        result = (in_range, inclusive, x_pf)

    fn dot_bounds_x[x_halfspace: Int, rounding: Int](
        self,
        group: Self.Group[rounding=rounding],
        y_pf: SIMD[dtype,simd_width],
        out dot_bounds_x: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]]
    ):
        # compute bounds on the dot product of the x coords with the plane normals,
        # for in-range testing
        var dot = Vec[3](
            x=group.normal_p[0]().y(),
            y=group.normal_p[1]().y(),
            z=group.normal_p[2]().y()
        )*y_pf
        var dot_rounded = dot.round[rounding]()
        dot_bounds_x = self.get_seg_bounds_vf[x_halfspace]()
        dot_bounds_x[0] -= dot_rounded
        dot_bounds_x[1] -= dot_rounded

    fn bound_fx_pf[
        x_halfspace: Int,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group,
        y_pi: SIMDInt[simd_width],
        out bound_fx_pf: _PBound[1,dtype,simd_width]
    ):
        var y_pf = SIMD[dtype,simd_width](y_pi)
        var dot_bounds_x = self.dot_bounds_x[x_halfspace](group, y_pf)

        ref pxy = group.planes_xy
        ref pyz = group.planes_yz
        ref pzx = group.planes_zx

        bound_fx_pf = _PBound[1,dtype,simd_width]()

        @parameter
        for i in [0,1]:
            bound_fx_pf.update(self.intersect_x[x_halfspace,i,debug=debug,debugger=debugger](group, pxy, y_pf, dot_bounds_x))
            bound_fx_pf.update(self.intersect_x[x_halfspace,i,debug=debug,debugger=debugger](group, pyz, y_pf, dot_bounds_x))
            bound_fx_pf.update(self.intersect_x[x_halfspace,i,debug=debug,debugger=debugger](group, pzx, y_pf, dot_bounds_x))

        @parameter
        if debug:
            ref dbg = debugger()[]
            var _w = dbg.projection_offset(group)
            if _w is not None:
                var w = _w.value()
                dbg.log(String("bound_fx_pf()",
                    "  x_halfspace=", x_halfspace,
                    "  y_pf=", y_pf[w],
                    "  bound_fx_pf=", bound_fx_pf.f[slice=w],
                ))

    # slow path: only use in testing
    fn bound_fx_pf[
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        group: Self.Group,
        y_pi: SIMDInt[simd_width],
        out bound_fx_pf: _PBound[1,dtype,simd_width],
        *,
        x_halfspace: Int
    ):
        if x_halfspace == 1:
            bound_fx_pf = self.bound_fx_pf[1,debug=debug,debugger=debugger](group, y_pi)
        else:
            bound_fx_pf = self.bound_fx_pf[-1,debug=debug,debugger=debugger](group, y_pi)


@fieldwise_init
struct _Intersection[dim: Int, dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var in_range: SIMDBool[simd_width]
    var point: Vec[dim,SIMD[dtype,simd_width]]


@fieldwise_init
struct _PBound[dim: Int, dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var mask: SIMDBool[simd_width]
    var f: _Bounds[dim,dtype,simd_width]

    fn __init__(out self):
        self.mask = SIMDBool[simd_width](fill=False)
        self.f = _Bounds[dim,dtype,simd_width]()

    @always_inline
    fn select[d: Int](self, out result: _PBound[1,dtype,simd_width]):
        result = _PBound[1,dtype,simd_width](
            mask = self.mask,
            f = self.f.select[d]()
        )

    @always_inline
    fn all_empty(self) -> Bool:
        return not self.mask.reduce_or() or self.f.all_empty()

    @always_inline
    fn is_empty(self, w: Int) -> Bool:
        return not self.mask[w] or self.f.is_empty(w)

    @always_inline
    fn update(
        mut self,
        in_range: SIMDBool[simd_width]
    ):
        self.mask |= in_range

    @always_inline
    fn update(
        mut self,
        in_range: SIMDBool[simd_width],
        f: Vec[dim,SIMD[dtype,simd_width]]
    ):
        self.f.update(in_range, f, self.mask)
        self.update(in_range)

    @always_inline
    fn update(
        mut self,
        in_range: SIMDBool[simd_width],
        f: Vec[dim,SIMD[dtype,simd_width]],
        inclusive: SIMDBool[simd_width]
    ):
        self.f.update(in_range, f, self.mask, inclusive)
        self.update(in_range)

    @always_inline
    fn update(
        mut self: _PBound[1,dtype,1],
        values: Tuple[Bool,Bool,Scalar[dtype]]
    ):
        var (in_range, inclusive, f_pf) = values
        self.update((
            SIMDBool[1](fill=in_range),
            SIMDBool[1](fill=inclusive),
            SIMD[dtype,1](f_pf)
        ))

    @always_inline
    fn update(
        mut self: _PBound[1,dtype,simd_width],
        values: Tuple[SIMDBool[simd_width],SIMDBool[simd_width],SIMD[dtype,simd_width]]
    ):
        var (in_range, inclusive, f_pf) = values
        self.update(in_range, Vec[1](x=f_pf), inclusive)

    # TODO: get rid of any unused update functions?

    @always_inline
    fn update(
        mut self,
        i: _Intersection[dim,dtype,simd_width]
    ):
        self.update(i.in_range, i.point)

    fn __getitem__(self, *, slice: Int, out result: _PBound[dim,dtype,1]):
        result = _PBound[dim,dtype,1](
            mask = self.mask[slice],
            f = self.f[slice=slice]
        )


@fieldwise_init
struct _Bounds[dim: Int, dtype: DType, simd_width: Int](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var min: Vec[dim,SIMD[dtype,simd_width]]
    var max: Vec[dim,SIMD[dtype,simd_width]]
    var min_inclusive: Vec[dim,SIMDBool[simd_width]]
    var max_inclusive: Vec[dim,SIMDBool[simd_width]]

    fn __init__(out self):
        # start with empty bounds
        self.min = Vec[dim,SIMD[dtype,simd_width]](fill=0)
        self.max = Vec[dim,SIMD[dtype,simd_width]](fill=0)
        self.min_inclusive = Vec[dim,SIMDBool[simd_width]](fill=False)
        self.max_inclusive = Vec[dim,SIMDBool[simd_width]](fill=False)

    fn set_inclusive(mut self, v: Bool):
        self.min_inclusive = Vec[dim](fill=SIMDBool[simd_width](fill=v))
        self.max_inclusive = Vec[dim](fill=SIMDBool[simd_width](fill=v))

    @always_inline
    fn update(
        mut self,
        in_range: SIMDBool[simd_width],
        v: Vec[dim,SIMD[dtype,simd_width]],
        has_value: SIMDBool[simd_width]
    ):
        var want_value = Vec[dim](fill=~has_value)
        var in_range_simd = Vec[dim](fill=in_range)
        var update_min = in_range_simd & (want_value | (v < self.min))
        var update_max = in_range_simd & (want_value | (v > self.max))
        self.min = update_min.select(v, self.min)
        self.max = update_max.select(v, self.max)

    @always_inline
    fn update(
        mut self,
        in_range: SIMDBool[simd_width],
        v: Vec[dim,SIMD[dtype,simd_width]],
        has_value: SIMDBool[simd_width],
        inclusive: SIMDBool[simd_width]
    ):
        @parameter
        for d in range(dim):

            # when the values are on the boundaries, update to inclusive when the values are inclusive
            # (but don't downgrade to exclusive)
            var mask_min = in_range & has_value & v[d].eq(self.min[d])
            var mask_max = in_range & has_value & v[d].eq(self.max[d])
            self.min_inclusive[d] |= mask_min.select(
                true_case = inclusive,
                false_case = SIMDBool[simd_width](fill=False)
            )
            self.max_inclusive[d] |= mask_max.select(
                true_case = inclusive,
                false_case = SIMDBool[simd_width](fill=False)
            )

            # when pusing out the boundaries, overwrite with new values
            mask_min = in_range & (~has_value | v[d].lt(self.min[d]))
            mask_max = in_range & (~has_value | v[d].gt(self.max[d]))
            self.min[d] = mask_min.select(v[d], self.min[d])
            self.max[d] = mask_max.select(v[d], self.max[d])
            self.min_inclusive[d] = mask_min.select(inclusive, self.min_inclusive[d])
            self.max_inclusive[d] = mask_max.select(inclusive, self.max_inclusive[d])

    @always_inline
    fn invert(mut self):
        swap(self.min, self.max)
        swap(self.min_inclusive, self.max_inclusive)
        self.min *= -1
        self.max *= -1
    
    @always_inline
    fn all_empty(self) -> Bool:
        return self.min.gt_any(self.max).reduce_and()

    @always_inline
    fn is_empty(self, w: Int) -> Bool:
        return self.min[slice=w].gt_any(self.max[slice=w])

    fn select[d: Int](self, out result: _Bounds[1,dtype,simd_width]):
        result = _Bounds[1,dtype,simd_width](
            min = self.min.select[d](),
            max = self.max.select[d](),
            min_inclusive = self.min_inclusive.select[d](),
            max_inclusive = self.max_inclusive.select[d]()
        )

    fn __getitem__(self, *, slice: Int, out result: _Bounds[dim,dtype,1]):
        result = _Bounds[dim,dtype,1](
            min = self.min[slice=slice],
            max = self.max[slice=slice],
            min_inclusive = self.min_inclusive[slice=slice],
            max_inclusive = self.max_inclusive[slice=slice]
        )

    fn write_to[W: Writer](self, mut writer: W):

        @parameter
        fn write_dim(v: SIMD[dtype,simd_width], inclusive: SIMDBool[simd_width]):
            @parameter
            for w in range(simd_width):
                @parameter
                if w > 0:
                    writer.write(":")
                writer.write(v[w])
                if inclusive[w]:
                    writer.write("*")

        @parameter
        for d in range(dim):
            @parameter
            if d > 0:
                writer.write("x")
            writer.write("[")
            write_dim(self.min[d], self.min_inclusive[d])
            writer.write(",")
            write_dim(self.max[d], self.max_inclusive[d])
            writer.write("]")

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct _SegmentNeighborhood[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var s00: Self.Segment
    var s10: Self.Segment
    var s01: Self.Segment
    var s11: Self.Segment
    var in_range_x_mask: SIMDBool[simd_width]

    comptime Segment = ComplexSIMD[dtype,simd_width]
    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()

    fn voxel_neighborhood[x_halfspace: Int, out_of_range: OutOfRangeBehavior[dtype]](
        self,
        x_offset: Int,
        out voxel_neighborhood: _VoxelNeighborhood[dtype]
    ):

        # reverse the offset, if needed
        var i = x_offset
        @parameter
        if x_halfspace == -1:
            i = self.num_neighborhoods_in_segment - i - 1

        # apply out-of-range behavior
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not self.in_range_x_mask[i]:
                voxel_neighborhood = _neighborhood_out_of_range[out_of_range]()
                return

        voxel_neighborhood = complex.pack[8](
            complex.slice[2](self.s00, i),
            complex.slice[2](self.s10, i),
            complex.slice[2](self.s01, i),
            complex.slice[2](self.s11, i)
        )


fn _neighborhood_out_of_range[
    dtype: DType,
    //,
    out_of_range: OutOfRangeBehavior[dtype]
](out neighborhood: _VoxelNeighborhood[dtype]):
    neighborhood = complex.pack[8](
        out_of_range.value,
        out_of_range.value,
        out_of_range.value,
        out_of_range.value,
        out_of_range.value,
        out_of_range.value,
        out_of_range.value,
        out_of_range.value
    )


fn calc_f_vi[x_halfspace: Int](f_vi_pos: Vec[3,Int], out f_vi: Vec[3,Int]):
    f_vi = f_vi_pos.copy()
    @parameter
    if x_halfspace == -1:
        # in the negative halfspace, apply the inversion symmetry,
        # then subtract one to move to the origin coordinates of that voxel
        f_vi = -f_vi - 1


fn calc_f_vi_corner[x_halfspace: Int, simd_width: Int](f_vi: Vec[3,Int], out f_vi_corner: Vec[3,Int]):
    f_vi_corner = f_vi.copy()
    @parameter
    if x_halfspace == -1:
        # in the negative halfspace, the corner of the segment is at the far end
        f_vi_corner.x() -= _num_neighborhoods_in_segment[simd_width]() - 1


# complex value rendering code, mostly only useful for debugging

fn _render_i[dtype: DType](i: Scalar[dtype], out s: String):
    # values are triples of single digits
    s = String(Int(i))
    # but get rid of the sign, if any
    if i < 0:
        s = s[1:]
    # but drop the z coordinate (it's usually redundant for testing)
    s = s[:len(s) - 1]
    # pad to 2 characters
    while len(s) < 2:
        s = "0" + s


fn _render_v[dtype: DType](v: ComplexScalar[dtype]) -> String:
    var re = _render_i(v.re)
    var im = _render_i(v.im)
    if v.im >= 0:
        return String(re, "+", im, "i")
    else:
        return String(re, "-", im, "i")


fn _render_segment[dtype: DType, simd_width: Int](segment: ComplexSIMD[dtype,simd_width], out s: String):
    s = "["
    @parameter
    for i in range(simd_width):
        s += "  "
        s += _render_v(complex.slice[i](segment))
    s += "  ]"


fn _render_neighborhood[dtype: DType](neighborhood: ComplexSIMD[dtype,8]) -> String:
    return String("[",
        "  ", _render_v(complex.slice[0](neighborhood)),
        "  ", _render_v(complex.slice[1](neighborhood)),
        "  ", _render_v(complex.slice[2](neighborhood)),
        "  ", _render_v(complex.slice[3](neighborhood)),
        # NOTE: just show the first four parts (z_0),
        #       since the last four (z_1) are usually redundant for testing
    "  ]")


fn _render_neighborhood[dtype: DType, simd_width: Int](neighborhood: _SegmentNeighborhood[dtype,simd_width]) -> String:
    return String("SegmentNeighborhood[",
        "\n  00=", _render_segment(neighborhood.s00),
        "\n  10=", _render_segment(neighborhood.s10),
        "\n  01=", _render_segment(neighborhood.s01),
        "\n  11=", _render_segment(neighborhood.s11),
        "\n   x=", neighborhood.in_range_x_mask,
    "\n]")


struct ScanDebugger(
    Copyable,
    Movable
):
    var proj_i: Int
    var f_pi: Vec[2,Int]  # projection sample coords
    var f_vi: Vec[3,Int]  # segment coords

    var msgs: List[String]

    fn __init__(
        out self,
        proj_i: Int,
        f_pi: Vec[2,Int],
        f_vi: Vec[3,Int]
    ):
        self.proj_i = proj_i
        self.f_pi = f_pi.copy()
        self.f_vi = f_vi.copy()

        self.msgs = List[String]()

    fn is_projection(self, proj_group: _ProjectionGroup[_,_,rounding=_], w: Int) -> Bool:
        return proj_group.proj_indices[w] == self.proj_i

    fn projection_coords[simd_width: Int](self, projections: _Projections[_,simd_width,_,rounding=_]) -> Tuple[Int,Int]:
        for g in range(projections.num_groups):
            var w = self.projection_offset(projections.groups[g])
            if w is not None:
                return (g, w.value())
        return abort[Tuple[Int,Int]]("Target projection not found")

    fn projection_offset[simd_width: Int](self, proj_group: _ProjectionGroup[_,simd_width,rounding=_]) -> Optional[Int]:

        # TEMP: needed to avoid a miscompilation bug with closures =(
        var num_projections = proj_group.num_projections
        if num_projections > simd_width:
            print("WARNING: Miscompilation broke num_projections somehow:", num_projections,
                "\n\tSetting num_projections to simd_width instead"
            )
            num_projections = simd_width

        for w in range(num_projections):
            if self.is_projection(proj_group, w):
                return w
        return None

    fn is_segment[x_halfspace: Int, simd_width: Int](self, f_vi: Vec[3,Int]) -> Bool:
        @parameter
        for n in range(_num_neighborhoods_in_segment[simd_width]()):
            var f_vi_vox = f_vi + Vec[3](x=x_halfspace*n, y=0, z=0)
            if f_vi_vox == self.f_vi:
                return True
        return False

    fn is_segment_xz[x_halfspace: Int, simd_width: Int](self, x_vi: Int, z_vi: Int) -> Bool:
        var f_vi = Vec[3](x=x_vi, y=self.f_vi.y(), z=z_vi)
        return self.is_segment[x_halfspace,simd_width](f_vi)

    fn is_projection_segment[x_halfspace: Int, simd_width: Int](
        self,
        proj_group: _ProjectionGroup[_,_,rounding=_],
        w: Int,
        f_vi: Vec[3,Int]
    ) -> Bool:
        return self.is_projection(proj_group, w)
            and self.is_segment[x_halfspace,simd_width](f_vi)

    fn is_sample(self, f_pi: Vec[2,Int]) -> Bool:
        return self.f_pi == f_pi

    fn is_projection_sample(
        self,
        proj_group: _ProjectionGroup[_,_,rounding=_],
        w: Int,
        f_pi: Vec[2,Int]
    ) -> Bool:
        return self.is_projection(proj_group, w)
            and self.is_sample(f_pi)

    fn log(mut self, msg: String):
        self.msgs.append(msg)


@parameter
fn _no_debugger() -> UnsafePointer[ScanDebugger,MutAnyOrigin]:
    return UnsafePointer[ScanDebugger,MutAnyOrigin]()
    # yes, deliberately return a null pointer
    # the caller should (hopefully) never actually dereference it
