
from math import floor, ceil, ceildiv
from complex import ComplexSIMD
from utils.numerics import inf

from cryoluge.collections import MovableList
from cryoluge.math import Vec, AlignedBox, OrientedBox, complex, ladder
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
        var p = Profiler(unit='us')
        p.start('scan')

        var coords_proj = FFTCoords(sizes_real_proj)
        var freq_limits_proj = freq_limits.checker(sizes_real_proj)

        p.start('projections')  # TEMP
        var simd_projections = _Projections[simd_width,rounding=rounding](projections)
        # TODO: is this a heap allocation? can we move it outside the function somehow?
        p.stop('projections')  # TEMP

        # iterate over the voxel coords that cover the projection range
        p.start('v-bounds')  # TEMP
        var bounds = self._voxels_bounds[rounding=rounding](coords_proj, projections)
        ref f_v_mini = bounds[0]
        ref f_v_maxi = bounds[1]
        p.stop('v-bounds')  # TEMP

        # TEMP
        ref p_p_compute = p.counter('p_compute')
        ref p_p_advance = p.counter('p_advance')
        ref p_p_bounds_f = p.counter('p_bounds_f')
        ref p_p_bounds_i = p.counter('p_bounds_i')
        ref p_xp_bounds = p.counter('xp_bounds')
        ref p_samples = p.counter('samples')
        ref p_s_rot = p.counter('s_rot')
        ref p_s_dists = p.counter('s_dists')
        ref p_s_freqs = p.counter('s_freqs')
        ref p_s_segments = p.counter('s_segments')
        ref p_s_neighborhood = p.counter('s_neighborhood')
        ref p_s_interp = p.counter('s_interp')
        ref p_s_func = p.counter('s_func')

        # TEMP
        var samples_tested = 0
        var samples_accepted = 0
        var scanlines_tested = 0
        var scanlines_accepted = 0

        comptime segment_sizes = _ProjectionGroup[dtype,simd_width].segment_sizes

        # TODO: need to clean up this function a bit

        # allocate space for intersection information
        # TODO: can we combine this with the other projection info?
        var intersection_groups = MovableList[_XHalfspaces[_PIntersections[dtype,simd_width]]](capacity=len(simd_projections.groups))
        for _ in range(len(simd_projections.groups)):
            intersection_groups.append(_XHalfspaces(
                pos = _PIntersections[dtype,simd_width](),
                neg = _PIntersections[dtype,simd_width]()
            ))

        for x in range(f_v_mini.x(), f_v_maxi.x() + 1, Self.num_neighborhoods_in_segment):
            for z in range(f_v_mini.z(), f_v_maxi.z() + 1):

                # compute all the intersections for this z-line
                # start below the y-min
                p_p_compute.start()  # TEMP
                var f_vi_pos = Vec[3](x=x, y=f_v_mini.y() - 1, z=z)
                for g in range(len(simd_projections.groups)):
                    ref proj_group = simd_projections.groups[g]

                    # TODO: these coordinate transforms are duplicated, can we consolidate?
                    @parameter
                    for x_halfspace in [1, -1]:

                        var f_vi = f_vi_pos.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi = -f_vi - 1

                        var f_vi_corner = f_vi.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi_corner.x() -= materialize[segment_sizes.x() - 1]()

                        intersection_groups[g].get[x_halfspace]().compute(f_vi_corner, proj_group)
                p_p_compute.stop()  # TEMP

                for y in range(f_v_mini.y(), f_v_maxi.y() + 1):

                    var f_vi_pos = Vec[3](x=x, y=y, z=z)

                    # map to both positive and negative x halfspaces
                    @parameter
                    for x_halfspace in [1, -1]:

                        var f_vi = f_vi_pos.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi = -f_vi - 1

                        var f_vf = f_vi.map_scalar[dtype]()

                        # get the segment corner, which isn't necessarily in the origin voxel
                        var f_vi_corner = f_vi.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi_corner.x() -= materialize[segment_sizes.x() - 1]()

                        @parameter
                        if debug:
                            ref dbg = debugger()[]
                            if dbg.is_segment[x_halfspace,simd_width](f_vi):
                                dbg.log(String("iterating segment:",
                                    "  f_vi=", f_vi,
                                    "  x_halfspace=", x_halfspace,
                                    "  f_vi_corner=", f_vi_corner
                                ))

                        var segment_neighborhood: Optional[_SegmentNeighborhood[dtype,simd_width]] = None

                        # for each group of projections ...
                        for g in range(len(simd_projections.groups)):
                            ref proj_group = simd_projections.groups[g]

                            # advance the intersections along y
                            p_p_advance.start()  # TEMP
                            ref intersections = intersection_groups[g].get[x_halfspace]()
                            intersections.advance_y[x_halfspace](proj_group)
                            p_p_advance.stop()  # TEMP

                            # compute a y-bound on the intersection of the segement with the z_p=0 plane
                            p_p_bounds_f.start()  # TEMP
                            var bound_pf = intersections.bound_f(proj_group)
                            p_p_bounds_f.stop()  # TEMP

                            p_p_bounds_i.start()  # TEMP
                            var bound_pi = proj_group.bound_pi(bound_pf, coords_proj.fmin_pos(), coords_proj.fmax())
                            p_p_bounds_i.stop()  # TEMP

                            if bound_pi.all_empty():
                                continue

                            # for each projection in the group ...
                            for w in range(proj_group.num_projections):
                                ref proj = projections[proj_group.proj_indices[w]]

                                @parameter
                                if debug:
                                    ref dbg = debugger()[]
                                    if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                        dbg.log(String(
                                            "mask=", bound_pi.mask[w],
                                            "  bounds=", bound_pi.f[slice=w]
                                        ))
                                        dbg.log(proj_group.render_bound_geometry(w, x_halfspace, f_vi, f_vi_corner, proj, coords_proj))

                                # skip over empty bounds
                                if bound_pi.is_empty(w):
                                    continue
                                
                                p_samples.start()  # TEMP

                                # TEMP
                                var segment_samples_tested = 0
                                var segment_samples_accepted = 0
                                var segment_scanlines_tested = 0
                                var segment_scanlines_accepted = 0

                                # iterate over the projection sample lines in the y-bounds
                                for sy in range(bound_pi.f.min[1][w], bound_pi.f.max[1][w] + 1):

                                    # do intersection tests for each y-scanline to get tighter x-bounds
                                    p_xp_bounds.start()  # TEMP
                                    var bound_x_pf = proj_group.bound_x_pf[x_halfspace](f_vi_corner, Int(sy), proj)
                                    var bound_x_pi = proj_group.bound_pi(
                                        bound_x_pf,
                                        coords_proj.fmin_pos().select[0](),
                                        coords_proj.fmax().select[0]()
                                    )
                                    p_xp_bounds.stop()  # TEMP

                                    # TEMP
                                    segment_scanlines_tested += 1

                                    @parameter
                                    if debug:
                                        ref dbg = debugger()[]
                                        if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                            dbg.log(String(
                                                "y=", sy,
                                                "  mask=", bound_x_pi.mask[0],
                                                "  bound=", bound_x_pf.f[slice=0], " ", bound_x_pi.f[slice=0]
                                            ))
                                            _ = proj_group.bound_x_pf[x_halfspace, debug=True, debugger=debugger](f_vi_corner, Int(sy), proj)

                                    # TEMP
                                    if bound_x_pi.is_empty(0):
                                        continue
                                    segment_scanlines_accepted += 1

                                    for sx in range(bound_x_pi.f.min.x()[0], bound_x_pi.f.max.x()[0] + 1):
                                        var sf_pi = Vec[2](x=sx, y=sy).map_int()
                                        var sf_pf = sf_pi.map_scalar[dtype]()

                                        # TEMP
                                        segment_samples_tested += 1

                                        p_s_rot.start()  # TEMP
                                        
                                        # transform back into reference volume space
                                        var sf_vf = proj.proj_to_vol(sf_pf)
                                            .round[rounding]()

                                        # (double-)check if the sample point actually lies in the segment
                                        # (treat the upper boundaries as exclusive)
                                        var sf_vf_min = f_vi_corner.map_scalar[dtype]()
                                        var sf_vf_max = sf_vf_min + materialize[segment_sizes.map_scalar[dtype]()]()
                                        var in_bounds = sf_vf.ge_all(sf_vf_min) and sf_vf.lt_all(sf_vf_max)

                                        p_s_rot.stop()  # TEMP

                                        @parameter
                                        if debug:
                                            ref dbg = debugger()[]
                                            if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                                dbg.log(String(
                                                    "sf_pi=", sf_pi,
                                                    "  sv_vf=", sf_vf,
                                                    "  min=", sf_vf_min,
                                                    "  max=", sf_vf_max,
                                                    "  in_bounds=", in_bounds
                                                ))

                                        if not in_bounds:
                                            continue

                                        # TEMP
                                        segment_samples_accepted += 1

                                        # apply the frequency limits
                                        p_s_freqs.start()  # TEMP
                                        var in_freq = freq_limits_proj.contains(f=sf_pf)
                                        p_s_freqs.stop()  # TEMP
                                        if not in_freq:
                                            continue

                                        # load the segments, if needed
                                        if segment_neighborhood is None:
                                            p_s_segments.start()  # TEMP
                                            segment_neighborhood = self._segment_neighborhood[x_halfspace](f_vi_pos)
                                            p_s_segments.stop()  # TEMP

                                        p_s_neighborhood.start()  # TEMP

                                        # get the x-offset into the segment and the distances
                                        var x_offset = Int(floor(sf_vf.x())) - f_vi.x()
                                        @parameter
                                        if x_halfspace == -1:
                                            x_offset = -x_offset
                                        var dists_v = sf_vf - sf_vf.floor()

                                        @parameter
                                        if debug:
                                            ref dbg = debugger()[]
                                            if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                                dbg.log(String(
                                                    "  x_offset=", x_offset,
                                                    "  dists_v=", dists_v
                                                ))

                                        # finally, interpolate the reference volume
                                        var voxel_neighborhood = segment_neighborhood.value().voxel_neighborhood[x_halfspace, Self.out_of_range](x_offset)

                                        p_s_neighborhood.stop()  # TEMP

                                        p_s_interp.start()  # TEMP
                                        var sv = interpolate(dists_v, voxel_neighborhood)
                                        p_s_interp.stop()  # TEMP

                                        @parameter
                                        if debug:
                                            ref dbg = debugger()[]
                                            if dbg.is_projection_segment[x_halfspace,simd_width](proj_group, w, f_vi):
                                                dbg.log(String("segment_neighborhood=", _render_neighborhood(segment_neighborhood.value())))
                                                dbg.log(String("voxel_neighborhood=", _render_neighborhood(voxel_neighborhood)))
                                            if dbg.is_projection_sample(proj_group, w, sf_pi):
                                                dbg.log(String("sf_pi found in f_vi=", f_vi))

                                        p_s_func.start()  # TEMP
                                        func(proj.id, sf_pi^, sf_vf^, sv)
                                        p_s_func.stop()  # TEMP

                                p_samples.stop()  # TEMP

                                # TEMP
                                samples_tested += segment_samples_tested
                                samples_accepted += segment_samples_accepted
                                scanlines_tested += segment_scanlines_tested
                                scanlines_accepted += segment_scanlines_accepted
                                # TODO: look at the scanlines we rejected, see what's going on there
                                #       especially if one segment rejects a lot of them

                            # end of projection loop
                        # end of projection group loop
                    # end of halfspace loop
                # end of voxel loop

        # TEMP
        p.stop('scan')
        print(p)
        print("report:"
            "  samples:",
            " tested=", samples_tested,
            ", accepted=", samples_accepted,
            " (", samples_accepted*100.0/samples_tested, "%)",
            "  scanlines:",
            " tested=", scanlines_tested,
            ", accepted=", scanlines_accepted,
            " (", scanlines_accepted*100.0/scanlines_tested, "%)",
            sep=""
        )


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
    var rot_proj_to_vol: Matrix[3,3,dtype,simd_width]
    var segment_extents_neg: Vec[3,SIMD[dtype,simd_width]]
    var segment_extents_pos: Vec[3,SIMD[dtype,simd_width]]
    var planes_xy: Self.Planes[Self.PX,Self.PY,Self.PZ]
    var planes_xz: Self.Planes[Self.PX,Self.PZ,Self.PY]
    var planes_yz: Self.Planes[Self.PY,Self.PZ,Self.PX]

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
        self.rot_proj_to_vol = Matrix[3,3,dtype,simd_width](fill=0)
        self.segment_extents_neg = Vec[3](fill=zero_f)
        self.segment_extents_pos = Vec[3](fill=zero_f)
        self.planes_xy = Self.Planes[Self.PX,Self.PY,Self.PZ]()
        self.planes_xz = Self.Planes[Self.PX,Self.PZ,Self.PY]()
        self.planes_yz = Self.Planes[Self.PY,Self.PZ,Self.PX]()

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
    fn normal_p[p: _Plane](
        self,
        out normal_p: Vec[3,SIMD[dtype,simd_width]]
    ):
        normal_p = self.rot_proj_to_vol.vec(row=p.d)

    @always_inline
    fn normal_v[p: _Plane](
        self,
        out normal_v: Vec[3,SIMD[dtype,simd_width]]
    ):
        normal_v = self.rot_proj_to_vol.vec(col=p.d)

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

    fn intersect_x_p[
        d: Int,
        l: Int,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        y_pi: Int,
        f_vi_corner: Vec[3,Int],
        proj: VolumeNeighborhoodsProjection[dtype],
        mut in_range: Bool,
        mut inclusive: Bool,
        mut f_pf: Vec[1,Scalar[dtype]]
    ):
        # define the projection-space line using a paramaetric equation
        var p = proj.proj_to_vol(Vec[3](x=0, y=y_pi, z=0).map_scalar[dtype]())
        var n = proj.proj_to_vol(Vec[3](x=1, y=0, z=0).map_scalar[dtype]())

        # solve for the independent parameter of the intersection
        comptime ld = l*Self.segment_sizes[d]
        var vd = Scalar[dtype]( ld + f_vi_corner[d] )
        t = (vd - p[d])/n[d]
        
        # build the intersection point from the parametric line equation
        f_vf = p + n*t

        # determine if the point lies within the segment
        var f_vf_min = f_vi_corner.map_scalar[dtype]()
        var f_vf_max = f_vf_min + materialize[Self.segment_sizes.map_scalar[dtype]()]()
        var f_vf_rounded = f_vf.round[rounding]()
        in_range = f_vf_rounded.ge_all(f_vf_min) and f_vf_rounded.le_all(f_vf_max)

        # determine inclusivity (upper segment boundaries are exclusive)
        inclusive = f_vf.ne_all(f_vf_max)

        # rotate into projection space
        f_pf = proj.vol_to_proj(f_vf)
            .project[1]()
        
        @parameter
        if debug:
            ref dbg = debugger()[]
            dbg.log(String("intersect_x_p:",
                "  d=", d,
                "  l=", l,
                "  f_vf=", f_vf,
                "  f_vf_min=", f_vf_min,
                "  f_vf_max=", f_vf_max
            ))

    fn bound_x_pf[
        x_halfspace: Int,
        *,
        debug: Bool = False,
        debugger: fn () capturing -> UnsafePointer[ScanDebugger,MutAnyOrigin] = _no_debugger
    ](
        self,
        f_vi_corner: Vec[3,Int],
        y_pi: Int,
        proj: VolumeNeighborhoodsProjection[dtype],
        out bound_x_pf: _PBound[1,dtype,1]
    ):
        bound_x_pf = _PBound[1,dtype,1]()

        @parameter
        for d in range(3):
            @parameter
            for l in [0,1]:

                var in_range = False
                var inclusive = False
                var f_pf = Vec[1,Scalar[dtype]](fill=0)
                self.intersect_x_p[d,l,debug=debug,debugger=debugger](
                    y_pi, f_vi_corner, proj,
                    in_range, inclusive, f_pf
                )

                f_pf = f_pf.round[rounding]()

                bound_x_pf.update(
                    SIMDBool[1](fill=in_range),
                    f_pf,
                    SIMDBool[1](fill=inclusive)
                )

                @parameter
                if debug:
                    ref dbg = debugger()[]
                    dbg.log(String("bound_x_pf:",
                        "  d=", d,
                        "  l=", l,
                        "  in_range=", in_range,
                        "  inclusive=", inclusive,
                        "  f_pf=", f_pf,
                        "  bound_x_pf=", bound_x_pf.f[slice=0]
                    ))

    # for debugging
    fn render_bound_geometry(
        self,
        w: Int,
        x_halfspace: Int,
        f_vi: Vec[3,Int],
        f_vi_corner: Vec[3,Int],
        proj: VolumeNeighborhoodsProjection[dtype],
        coords_proj: FFTCoords[2],
        out str: String
    ):
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
            var p_pf = intersections.intersection[uv](planes)[slice=w]
            if intersections.in_range[uv](planes)[w]:
                inside_points.append(p_pf^)
            else:
                outside_points.append(p_pf^)

        # collect all 12 intersection points
        @parameter
        for uv in _UV.all:
            classify_intersection[uv](self.planes_xy)
            classify_intersection[uv](self.planes_xz)
            classify_intersection[uv](self.planes_yz)

        # compute the p-bounds
        var bound_pf = intersections.bound_f(self)
        var bound_pi = self.bound_pi(bound_pf, coords_proj.fmin_pos(), coords_proj.fmax())

        # compute the x p-bounds
        var x_in = List[Vec[2,Scalar[dtype]]]()
        var x_out = List[Vec[2,Scalar[dtype]]]()
        for fy_pi in range(Int(bound_pi.f.min[1][w]), Int(bound_pi.f.max[1][w]) + 1):
            var fy_pf = Scalar[dtype](fy_pi)

            @parameter
            for d in range(3):
                @parameter
                for l in [0,1]:
                    var in_range = False
                    var inclusive = False
                    var f_pf = Vec[1,Scalar[dtype]](x=0)
                    self.intersect_x_p[d,l](fy_pi, f_vi_corner, proj, in_range, inclusive, f_pf)

                    if in_range:
                        x_in.append(f_pf.lift(y=fy_pf))
                    else:
                        x_out.append(f_pf.lift(y=fy_pf))

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
            "\ngrid_p=[", coords_proj.fmin_pos(), ",", coords_proj.fmax(), "]",
            "\nf_pf=pt", self.vol_to_proj(f_vi.map_scalar[dtype]())[slice=w],
            "\nf_pf_corner=pt", self.vol_to_proj(f_vf_corner)[slice=w],
            "\naxes=[",
                "\n\tpt", proj.vol_to_proj(Vec[3](x=1, y=0, z=0).map_scalar[dtype]()), ","
                "\n\tpt", proj.vol_to_proj(Vec[3](x=0, y=1, z=0).map_scalar[dtype]()), ","
                "\n\tpt", proj.vol_to_proj(Vec[3](x=0, y=0, z=1).map_scalar[dtype]()),
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
            "\n]"
        )


struct _Projections[simd_width: Int, dtype: DType, *, rounding: Int = 0](
    Copyable,
    Movable
):
    var groups: List[Self.Group]

    comptime Group = _ProjectionGroup[dtype,simd_width,rounding=rounding]

    fn __init__(out self, projections: List[VolumeNeighborhoodsProjection[dtype]]):

        # allocate all the groups
        var num_groups = ceildiv(len(projections), simd_width)
        self.groups = List(length=num_groups, fill=Self.Group())

        # populate the groups with each projection
        for p in range(len(projections)):
            ref proj = projections[p]
            var g = p // simd_width
            ref group = self.groups[g]
            var i = p % simd_width

            group.num_projections += 1
            group.proj_indices[i] = p

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
            g.planes_xy = _Planes[dtype,simd_width,Self.Group.PX,Self.Group.PY,Self.Group.PZ](g)
            g.planes_xz = _Planes[dtype,simd_width,Self.Group.PX,Self.Group.PZ,Self.Group.PY](g)
            g.planes_yz = _Planes[dtype,simd_width,Self.Group.PY,Self.Group.PZ,Self.Group.PX](g)


@fieldwise_init
struct _UV(
    ImplicitlyCopyable,
    Movable
):
    var u: Int
    var v: Int

    comptime all = [_UV(0,0), _UV(0,1), _UV(1,0), _UV(1,1)]


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
    Movable
):
    # TODO: rename these to something more meaningful!
    var f: Vec[2,SIMD[dtype,simd_width]]
    var s1: Vec[3,SIMD[dtype,simd_width]]
    var s2: Vec[3,SIMD[dtype,simd_width]]
    var u: Vec[2,SIMD[dtype,simd_width]]
    var v: Vec[2,SIMD[dtype,simd_width]]

    fn __init__(out self):
        self.f = Vec[2,SIMD[dtype,simd_width]](fill=0)
        self.s1 = Vec[3,SIMD[dtype,simd_width]](fill=0)
        self.s2 = Vec[3,SIMD[dtype,simd_width]](fill=0)
        self.u = Vec[2,SIMD[dtype,simd_width]](fill=0)
        self.v = Vec[2,SIMD[dtype,simd_width]](fill=0)

    fn __init__(
        out self,
        group: _ProjectionGroup[dtype,simd_width,rounding=_]
    ):
        # intersection forumla for two axis-aligned volume-space planes with z_p=0,
        # but only the third coordinate
        # ie, at a point p_v, the third coord is p_v.f
        var unit_z_v = group.normal_v[_Plane.z()]()
        self.f = -Vec[2](x=unit_z_v[p1.d], y=unit_z_v[p2.d])/unit_z_v[p3.d]

        # get the plane normals and mix them
        var n0 = group.normal_p[p1]()
        var n1 = group.normal_p[p2]()
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
        self.u = Vec[2](x=n00, y=n01).two_inner_products(u1, u2)*p1.len
        self.v = Vec[2](x=n01, y=n11).two_inner_products(u1, u2)*p2.len

    @always_inline
    fn project(
        self,
        v: Vec[3,SIMD[dtype,simd_width]],
        out result: Vec[2,SIMD[dtype,simd_width]]
    ):
        result = Vec[2](x=v[p1.d], y=v[p2.d])

    @always_inline
    fn d3_vf_terms(
        self,
        seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]],
        out d3_vf_terms: Tuple[Vec[2,SIMD[dtype,simd_width]],Vec[2,SIMD[dtype,simd_width]]]
    ):
        d3_vf_terms = (
            self.project(seg_bounds_vf[0])*self.f,
            self.project(seg_bounds_vf[1])*self.f
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


@fieldwise_init
struct _XHalfspaces[T: AnyType & Copyable & Movable](
    Copyable,
    Movable
):
    var pos: T
    var neg: T

    fn get[x_halfspace: Int](ref self) -> ref [self.pos, self.neg] T:
        @parameter
        if x_halfspace == 1:
            return self.pos
        else:
            return self.neg


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
    var seg_bounds_vf: Tuple[Vec[3,SIMD[dtype,simd_width]],Vec[3,SIMD[dtype,simd_width]]]
    var d3_vf_terms: _PlaneMap[3,Tuple[Vec[2,SIMD[dtype,simd_width]],Vec[2,SIMD[dtype,simd_width]]]]
    var points_pf: _PlaneMap[3,Vec[2,SIMD[dtype,simd_width]]]

    comptime Group = _ProjectionGroup[dtype,simd_width,rounding=_]
    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()
    comptime segment_sizes = Vec[3,Int](x=Self.num_neighborhoods_in_segment, y=1, z=1)

    fn __init__(out self):

        comptime zero = SIMD[dtype,simd_width](0)
        comptime zero2 = Vec[2,SIMD[dtype,simd_width]](fill=0)
        comptime zero3 = Vec[3,SIMD[dtype,simd_width]](fill=0)

        self.seg_bounds_vf = materialize[(zero3, zero3)]()
        self.d3_vf_terms = _PlaneMap[3](fill=materialize[(zero2, zero2)]())
        self.points_pf = _PlaneMap[3](fill=materialize[zero2]())

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
        ref pxz = group.planes_xz
        ref pyz = group.planes_yz

        # compute the intersection point third dimension coordinates in volume-space, for in-range testing
        self.d3_vf_terms[pxy] = pxy.d3_vf_terms(self.seg_bounds_vf)
        self.d3_vf_terms[pxz] = pxz.d3_vf_terms(self.seg_bounds_vf)
        self.d3_vf_terms[pyz] = pyz.d3_vf_terms(self.seg_bounds_vf)

        # compute the intersection points themselves
        self.points_pf[pxy] = pxy.intersect_min_p(self.seg_bounds_vf)
        self.points_pf[pxz] = pxz.intersect_min_p(self.seg_bounds_vf)
        self.points_pf[pyz] = pyz.intersect_min_p(self.seg_bounds_vf)

    @always_inline
    fn in_range[uv: _UV, p3: _Plane](
        self,
        planes: _Planes[dtype,simd_width,_,_,p3],
        out in_range: SIMDBool[simd_width]
    ):
        var c3 = self.d3_vf_terms[planes][uv.u][0] + self.d3_vf_terms[planes][uv.v][1]
        in_range = c3.ge(self.seg_bounds_vf[0][p3.d])
            .__and__(c3.le(self.seg_bounds_vf[1][p3.d]))

    @always_inline
    fn intersection[uv: _UV, p3: _Plane](
        self,
        planes: _Planes[dtype,simd_width,_,_,p3],
        out i_pf: Vec[2,SIMD[dtype,simd_width]]
    ):
        i_pf = self.points_pf[planes] + planes.u*uv.u + planes.v*uv.v

    @always_inline
    fn advance_y[dy: Int](
        mut self,
        group: Self.Group
    ):
        # update the segment bounds
        self.seg_bounds_vf[0].y() += dy
        self.seg_bounds_vf[1].y() += dy

        ref pxy = group.planes_xy
        ref pxz = group.planes_xz
        ref pyz = group.planes_yz

        # advance only the intersection volume-space bounds that are affected by y_v
        self.d3_vf_terms[pxy][0][1] += pxy.f[1]*dy
        self.d3_vf_terms[pxy][1][1] += pxy.f[1]*dy
        self.d3_vf_terms[pyz][0][0] += pyz.f[0]*dy
        self.d3_vf_terms[pyz][1][0] += pyz.f[0]*dy

        # advance the intersection points
        self.points_pf[pxy] += pxy.sy()*dy
        self.points_pf[pxz] += pxz.sy()*dy
        self.points_pf[pyz] += pyz.sy()*dy

    @always_inline
    fn bound_f(
        self,
        group: Self.Group,
        out bound_pf: _PBound[2,dtype,simd_width]
    ):
        ref pxy = group.planes_xy
        ref pxz = group.planes_xz
        ref pyz = group.planes_yz

        bound_pf = _PBound[2,dtype,simd_width]()
        @parameter
        for uv in _UV.all:
            bound_pf.update(self.in_range[uv](pxy), self.intersection[uv](pxy))
            bound_pf.update(self.in_range[uv](pxz), self.intersection[uv](pxz))
            bound_pf.update(self.in_range[uv](pyz), self.intersection[uv](pyz))

        # start with the bounds being inclusive by default
        bound_pf.f.set_inclusive(True)


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
        self.min = update_min.select(
            true_case = v,
            false_case = self.min
        )
        self.max = update_max.select(
            true_case = v,
            false_case = self.max
        )

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
            self.min[d] = mask_min.select(
                true_case = v[d],
                false_case = self.min[d]
            )
            self.max[d] = mask_max.select(
                true_case = v[d],
                false_case = self.max[d]
            )
            self.min_inclusive[d] = mask_min.select(
                true_case = inclusive,
                false_case = self.min_inclusive[d]
            )
            self.max_inclusive[d] = mask_max.select(
                true_case = inclusive,
                false_case = self.max_inclusive[d]
            )

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

    fn is_segment[x_halfspace: Int, simd_width: Int](self, f_vi: Vec[3,Int]) -> Bool:
        @parameter
        for n in range(_num_neighborhoods_in_segment[simd_width]()):
            var f_vi_vox = f_vi + Vec[3](x=x_halfspace*n, y=0, z=0)
            if f_vi_vox == self.f_vi:
                return True
        return False

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
