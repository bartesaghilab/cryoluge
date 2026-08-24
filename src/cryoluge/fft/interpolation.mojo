
from math import floor, ceil, ceildiv
from complex import ComplexSIMD
from utils.numerics import inf

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
    *,
    dtype_coords: DType = dtype
](Movable):
    """
    A no-op implementation of the FFT interpolation, for testing,
    to see how well (or poorly) doing the interpolation with incoherent memory accesses really is.
    NOTE: It's very poor.
    """
    var _img: FFTImage[dim,dtype]
    var _out_of_range: OutOfRangeBehavior[dtype]

    comptime deltas = Delta[dim,dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[dtype,Self.num_samples]
    comptime EmptySamples[c: ComplexSIMD[dtype,1]] = ComplexSIMD[dtype,Self.num_samples](
        re=SIMD[dtype,Self.num_samples](c.re),
        im=SIMD[dtype,Self.num_samples](c.im)
    )

    fn __init__(
        out self,
        img: FFTImage[dim,dtype],
        out_of_range: OutOfRangeBehavior[dtype]
    ):
        self._img = img.copy()
        self._out_of_range = out_of_range

    fn get[
        simd_width: Int,
        *,
        or_else: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
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
                var v = self._img.get[or_else=or_else](f=f_sample)
                samples.re[s] = v.re
                samples.im[s] = v.im

                # TODO: handle out-of-range=override behavior

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

    fn _voxels_bounds(
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
        rounding: Optional[Int] = None
    ](
        self,
        sizes_real_proj: Vec[2,Int],
        projections: List[VolumeNeighborhoodsProjection[dtype]],
        freq_limits: FrequencyLimits[dtype] = FrequencyLimits[dtype].none()
    ):
        # TEMP
        from cryoluge.time import Profiler
        var p = Profiler(unit='us')
        p.start('scan')

        var coords_proj = FFTCoords(sizes_real_proj)
        var freq_limits_proj = freq_limits.checker(sizes_real_proj)

        p.start('projections')  # TEMP
        var simd_projections = _Projections[simd_width,rounding=rounding](projections)
        p.stop('projections')  # TEMP

        # iterate over the voxel coords that cover the projection range
        p.start('v-bounds')  # TEMP
        var bounds = self._voxels_bounds(coords_proj, projections)
        ref f_v_mini = bounds[0]
        ref f_v_maxi = bounds[1]
        p.stop('v-bounds')  # TEMP

        # TEMP
        ref p_extents = p.counter('extents')
        ref p_p_bounds = p.counter('p_bounds')
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

        for z in range(f_v_mini.z(), f_v_maxi.z() + 1):
            for y in range(f_v_mini.y(), f_v_maxi.y() + 1):
                for x in range(f_v_mini.x(), f_v_maxi.x() + 1, Self.num_neighborhoods_in_segment):

                    var f_vi_pos = Vec[3](x=x, y=y, z=z)

                    # map to both positive and negative x halfspaces
                    @parameter
                    for x_halfspace in [1, -1]:

                        var f_vi = f_vi_pos.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi = -f_vi - 1

                        var f_vf = f_vi.map_scalar[dtype]()

                        # TEMP
                        # var f_vi_focus = Vec[3](x=-1, y=-2, z=-1)
                        # var debug_v = False
                        # @parameter
                        # for w in range(Self.num_neighborhoods_in_segment):
                        #     var f_vi_w = f_vi + Vec[3](x=x_halfspace*w, y=0, z=0)
                        #     if f_vi_w == f_vi_focus:
                        #         debug_v = True
                        # if debug_v:
                        #     print("f_vi=", f_vi)

                        var segment_neighborhood: Optional[_SegmentNeighborhood[dtype,simd_width]] = None

                        # for each group of projections ...
                        for proj_group in simd_projections.groups:

                            var in_range = SIMDBool[simd_width](fill=True)
                            @parameter
                            fn any_in_range() -> Bool:
                                return in_range.reduce_or()

                            # TEMP: do a quick range check against the segment extents
                            # TODO: refactor this into the projection group?
                            # TODO: test this specifically ??
                            p_extents.start()  # TEMP
                            # var f_vi_corner = f_vi.copy()
                            # @parameter
                            # if x_halfspace == -1:
                            #     f_vi_corner.x() -= materialize[_ProjectionGroup[dtype,simd_width].PX.len - 1]()
                            # var f_pf = proj_group.vol_to_proj(f_vi_corner.map_scalar[dtype]())
                            # var f_pf_min = f_pf + proj_group.segment_extents_neg
                            # var f_pf_max = f_pf + proj_group.segment_extents_pos
                            # var f_pi_min = f_pf_min.ceil().map_dint()
                            # var f_pi_max = (f_pf_max - 1).ceil().map_dint()
                            #     .max(f_pi_min)

                            # var p_min = coords_proj.fmin_pos().lift(z=0).splat[simd_width]()
                            # var p_max = coords_proj.fmax().lift(z=0).splat[simd_width]()
                            # in_range = in_range
                            #     .__and__(f_pi_min.le_all(p_max))
                            #     .__and__(f_pi_max.ge_all(p_min))
                            p_extents.stop()  # TEMP
                            
                            if not any_in_range():
                                continue

                            # compute a bound on the intersection of the segement with the z_p=0 plane
                            p_p_bounds.start()  # TEMP
                            var bounds_p = proj_group.bound_pi(proj_group.bound_pf[x_halfspace](f_vi))
                            in_range = in_range.__and__(bounds_p.mask)
                            p_p_bounds.stop()  # TEMP

                            if not any_in_range():
                                continue

                            # for each projection in the group ...
                            for w in range(proj_group.num_projections):

                                if not in_range[w]:
                                    continue

                                ref proj = projections[proj_group.proj_indices[w]]

                                # TEMP
                                # if debug_v:
                                #     proj_group.print_intersection_geometry[x_halfspace](w, f_pf, proj, coords_proj)

                                p_samples.start()  # TEMP

                                # TEMP
                                var segment_samples_tested = 0
                                var segment_samples_accepted = 0

                                # iterate over the projection sample points in the bounding box
                                for sy in range(bounds_p.f.min.y()[w], bounds_p.f.max.y()[w] + 1):
                                    for sx in range(bounds_p.f.min.x()[w], bounds_p.f.max.x()[w] + 1):
                                        var sf_pi = Vec[2](x=sx, y=sy).map_int()
                                        var sf_pf = sf_pi.map_scalar[dtype]()

                                        # TEMP
                                        segment_samples_tested += 1

                                        # TEMP
                                        # if debug_v:
                                        #     print("\tsampling:", sf_pi)

                                        # TODO: NEXTTIME: there's a bug in here somewhere that the tests don't catch,
                                        #                 but csp2 shows wrong scores  =(
                                        #                 removing all segment bounds short-circuits doesn't fix it,
                                        #                 so the problem is probably something below?
                                        #                 need to get the tests to catch this bug!

                                        # transform back into reference volume space
                                        p_s_rot.start()  # TEMP
                                        var sf_vf = proj.proj_to_vol(sf_pf)
                                        p_s_rot.stop()  # TEMP

                                        # TODO: can get distances in projection-space too, right?

                                        # find out what voxel, if any, the point lies in
                                        # (treat the upper boundaries as exclusive)
                                        # and get its distances to the origin of that voxel
                                        p_s_dists.start()  # TEMP
                                        var dists_v: Vec[3,Scalar[dtype]]
                                        var x_offset: Int
                                        var in_bounds: Bool
                                        @parameter
                                        if x_halfspace == -1:
                                            var f_v_min = Vec[3](x=1 - Self.num_neighborhoods_in_segment, y=0, z=0).map_scalar[dtype]()
                                            var f_v_max = Vec[3](fill=1).map_scalar[dtype]()
                                            dists_v = sf_vf - f_vf
                                            in_bounds = dists_v.ge_all(f_v_min) and dists_v.lt_all(f_v_max)
                                            x_offset = Int(floor(dists_v.x()))
                                            dists_v.x() -= x_offset
                                            x_offset *= -1
                                        else:
                                            var f_v_min = Vec[3](fill=Scalar[dtype](0))
                                            var f_v_max = Vec[3](x=Self.num_neighborhoods_in_segment, y=1, z=1).map_scalar[dtype]()
                                            dists_v = sf_vf - f_vf
                                            in_bounds = dists_v.ge_all(f_v_min) and dists_v.lt_all(f_v_max)
                                            x_offset = Int(floor(dists_v.x()))
                                            dists_v.x() -= x_offset
                                        p_s_dists.stop()  # TEMP

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

                                        # finally, interpolate the reference volume
                                        p_s_neighborhood.start()  # TEMP
                                        var voxel_neighborhood = segment_neighborhood.value().voxel_neighborhood[x_halfspace, Self.out_of_range](x_offset)
                                        p_s_neighborhood.stop()  # TEMP
                                        p_s_interp.start()  # TEMP
                                        var sv = interpolate(dists_v, voxel_neighborhood)
                                        p_s_interp.stop()  # TEMP

                                        # TEMP
                                        # if debug_v:
                                        #     print("\t",
                                        #         "sample=", sf_pi,
                                        #         "x_offset=", x_offset,
                                        #         "dist_v_x=", dists_v.x()
                                        #     )
                                        #     print("\tsegment_neighborhood=", _render_neighborhood(segment_neighborhood.value()))
                                        #     print("\tvoxel_neighborhood=", _render_neighborhood(voxel_neighborhood))

                                        # TEMP
                                        # var debug_p = sf_pi == Vec[2](x=0, y=-2)
                                        # #var debug_p = False
                                        # if debug_p:
                                        #     print("\t\tsf_pi found in f_vi=", f_vi)

                                        p_s_func.start()  # TEMP
                                        func(proj.id, sf_pi^, sf_vf^, sv)
                                        p_s_func.stop()  # TEMP

                                p_samples.stop()  # TEMP

                                # TEMP
                                #print("\tsegment: tested=", segment_samples_tested, "accepted=", segment_samples_accepted)
                                samples_tested += segment_samples_tested
                                samples_accepted += segment_samples_accepted
                                # if segment_samples_accepted <= 0:
                                #     print("rejected all samples!")
                                #     print(proj_group.render_bound_geometry[x_halfspace](f_vi, proj))

        # TEMP
        p.stop('scan')
        print(p)
        print("samples:",
            " tested=", samples_tested,
            ", accepted=", samples_accepted,
            " (", samples_accepted*100.0/samples_tested, "%)",
            sep=""
        )
        # TODO: these acceptance percentages are still pretty low!


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
    fn proj_to_vol(self, v: Vec[3,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.rot_proj_to_vol*v

    @always_inline
    fn proj_to_vol(self, v: Vec[2,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.proj_to_vol(v.lift(z=0))

    @always_inline
    fn vol_to_proj[simd_width: Int](self, v: Vec[3,SIMD[dtype,simd_width]], out result: Vec[3,SIMD[dtype,simd_width]]):
        result = self.rot_proj_to_vol.mul_transpose(v)
    

comptime _VoxelNeighborhood[dtype: DType] = ComplexSIMD[dtype,8]


struct _ProjectionGroup[dtype: DType, simd_width: Int, *, rounding: Optional[Int] = None](
    Copyable,
    Movable
):
    var num_projections: Int
    var proj_indices: SIMDInt[simd_width]
    var vol_to_proj_xfactors: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_yfactors: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_zfactors: Vec[3,SIMD[dtype,simd_width]]
    var segment_extents_neg: Vec[3,SIMD[dtype,simd_width]]
    var segment_extents_pos: Vec[3,SIMD[dtype,simd_width]]
    var plane_x: Self.Plane[Self.PX]
    var plane_y: Self.Plane[Self.PY]
    var plane_z: Self.Plane[Self.PZ]
    var planes_xy: Self.Planes[Self.PX,Self.PY,Self.PZ]
    var planes_xz: Self.Planes[Self.PX,Self.PZ,Self.PY]
    var planes_yz: Self.Planes[Self.PY,Self.PZ,Self.PX]

    comptime num_neighborhoods_in_segment = _num_neighborhoods_in_segment[simd_width]()
    comptime PX = _CTPlane.x(Self.num_neighborhoods_in_segment)
    comptime PY = _CTPlane.y()
    comptime PZ = _CTPlane.z()
    comptime Plane = _RTPlane[dtype,simd_width,_]
    comptime Planes = _RTPlanes[dtype,simd_width,_]

    fn __init__(out self):
        comptime zero_i = SIMDInt[simd_width](0)
        comptime zero_f = SIMD[dtype,simd_width](0)
        self.num_projections = 0
        self.proj_indices = zero_i
        self.vol_to_proj_xfactors = Vec[3](fill=zero_f)
        self.vol_to_proj_yfactors = Vec[3](fill=zero_f)
        self.vol_to_proj_zfactors = Vec[3](fill=zero_f)
        self.segment_extents_neg = Vec[3](fill=zero_f)
        self.segment_extents_pos = Vec[3](fill=zero_f)
        self.plane_x = Self.Plane[Self.PX]()
        self.plane_y = Self.Plane[Self.PY]()
        self.plane_z = Self.Plane[Self.PZ]()
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
        # rotate the point(s) from volume space into projection space
        f_pf = Vec[3](
            x=self.vol_to_proj_xfactors.inner_product(f_vf),
            y=self.vol_to_proj_yfactors.inner_product(f_vf),
            z=self.vol_to_proj_zfactors.inner_product(f_vf)
        )

        # apply rounding behavior, if needed
        @parameter
        if Self.rounding is not None:
            f_pf = f_pf.__round__(Self.rounding.value())

    @always_inline
    fn intersect_v[d1: Int, d2: Int, p1: _CTPlane, p2: _CTPlane, p3: _CTPlane](
        self,
        f_vi: Vec[3,Int],
        planes: Self.Planes[p1,p2,p3],
        out f_vf: Vec[3,SIMD[dtype,simd_width]]
    ):
        # start at the segment position
        var c1 = f_vi[p1.d]
        var c2 = f_vi[p2.d]

        # offset by the two plane distances
        c1 += materialize[p1.len*d1]()
        c2 += materialize[p2.len*d2]()

        # get the third plane coordinate by intersecting with the z_p=0 plane
        # (for each projection)
        var c3 = Vec[2](x=c1, y=c2)
            .map_scalar[dtype]()
            .splat[simd_width]()
            .inner_product(planes.f)

        # build an intersection point for each projection
        f_vf = materialize[p1.normal_f[dtype]().splat[simd_width]()]()*c1
            + materialize[p2.normal_f[dtype]().splat[simd_width]()]()*c2
            + materialize[p3.normal_f[dtype]().splat[simd_width]()]()*c3

        # apply rounding behavior, if needed
        @parameter
        if Self.rounding is not None:
            f_vf = f_vf.__round__(Self.rounding.value())

    @always_inline
    fn in_range[p3: _CTPlane](
        self,
        f_vf: Vec[3,SIMD[dtype,simd_width]],
        mut in_range: SIMDBool[simd_width],
        mut inclusive_p3: SIMDBool[simd_width]
    ):
        var c = f_vf[p3.d]
        in_range = c.ge(0).__and__(c.le(p3.len))
        inclusive_p3 = c.ge(0).__and__(c.lt(p3.len))

    @always_inline
    fn intersect_p[d1: Int, d2: Int, p3: _CTPlane](
        self,
        f_vi: Vec[3,Int],
        planes: Self.Planes[_,_,p3],
        mut p_pf: Vec[2,SIMD[dtype,simd_width]],
        mut in_range: SIMDBool[simd_width],
        mut inclusive: SIMDBool[simd_width]
    ):
        # compute the intersection point
        var p_vf = self.intersect_v[d1,d2](f_vi, planes)
        p_pf = self.vol_to_proj(p_vf).project[2]()

        # check the range against the third planes
        var f_vf = f_vi.map_scalar[dtype]().splat[simd_width]()
        var inclusive_p3 = SIMDBool[simd_width](fill=False)
        self.in_range[p3](p_vf - f_vf, in_range, inclusive_p3)

        # calculate the inclusivity
        # ie, when the point lies on a segment boundary, is it an inclusive boundary
        @parameter
        if d1 == 0 and d2 == 0:
            inclusive = inclusive_p3
        else:
            inclusive = SIMDBool[simd_width](fill=False)

    # TODO: @always_inline?
    fn bound_pf[x_halfspace: Int](
        self,
        f_vi: Vec[3,Int],
        out bound_pf: _PBound[dtype,simd_width]
    ):
        # TEMP: move to the segment corner, if in the -x halfspace
        # TODO: share this somewhere?
        var f_vi_corner = f_vi.copy()
        @parameter
        if x_halfspace == -1:
            f_vi_corner.x() -= materialize[Self.PX.len - 1]()

        bound_pf = _PBound[dtype,simd_width]()

        @parameter
        @always_inline
        fn update[d1: Int, d2: Int, p1: _CTPlane, p2: _CTPlane, p3: _CTPlane](
            planes: Self.Planes[p1,p2,p3]
        ):
            # compute the intersection point
            var p_pf = Vec[2,SIMD[dtype,simd_width]](fill=0)
            var in_range = SIMDBool[simd_width](fill=False)
            var inclusive = SIMDBool[simd_width](fill=False)
            self.intersect_p[d1,d2](f_vi_corner, planes, p_pf, in_range, inclusive)

            # update the bounds
            bound_pf.update(in_range, inclusive, p_pf)

        # check all 12 intersection points
        update[0,0](self.planes_xy)
        update[0,1](self.planes_xy)
        update[1,0](self.planes_xy)
        update[1,1](self.planes_xy)

        update[0,0](self.planes_xz)
        update[0,1](self.planes_xz)
        update[1,0](self.planes_xz)
        update[1,1](self.planes_xz)

        update[0,0](self.planes_yz)
        update[0,1](self.planes_yz)
        update[1,0](self.planes_yz)
        update[1,1](self.planes_yz)

    # TODO: @always_inline?
    fn bound_pi(
        self,
        bound_pf: _PBound[dtype,simd_width],
        out bound_pi: _PBound[DType.int,simd_width]
    ):
        bound_pi = _PBound[DType.int,simd_width]()
        ref bf = bound_pf.f
        ref bi = bound_pi.f

        # the mask needs no changes
        bound_pi.mask = bound_pf.mask

        # discretize the bound, paying attention to the inclusivity of each coordinate
        @parameter
        for d in range(2):
            @parameter
            for w in range(simd_width):

                if bf.min_inclusive[d][w]:
                    bi.min[d][w] = SIMDInt[1]( ceil(bf.min[d][w]) )
                else:
                    bi.min[d][w] = SIMDInt[1]( floor(bf.min[d][w] + 1) )

                if bf.max_inclusive[d][w]:
                    bi.max[d][w] = SIMDInt[1]( floor(bf.max[d][w]) )
                else:
                    bi.max[d][w] = SIMDInt[1]( ceil(bf.max[d][w] - 1) )

        # the above logic creates fully-inclusive integer bounds
        bi.min_inclusive = Vec[2,SIMDBool[simd_width]](fill=SIMDBool[simd_width](fill=True))
        bi.max_inclusive = Vec[2,SIMDBool[simd_width]](fill=SIMDBool[simd_width](fill=True))

    # for debugging
    fn render_bound_geometry[x_halfspace: Int](
        self,
        w: Int,
        f_vi: Vec[3,Int],
        proj: VolumeNeighborhoodsProjection[dtype],
        out str: String
    ):
        # TEMP: move to the segment corner, if in the -x halfspace
        # TODO: share this somewhere?
        var f_vi_corner = f_vi.copy()
        @parameter
        if x_halfspace == -1:
            f_vi_corner.x() -= materialize[Self.PX.len - 1]()

        var f_vf_corner = f_vi_corner.map_scalar[dtype]().splat[simd_width]()

        # classify all the intersection points
        var inside_points = List[Tuple[Vec[2,Scalar[dtype]],String]]()
        var outside_points = List[Tuple[Vec[2,Scalar[dtype]],String]]()

        @parameter
        fn classify_intersection[d1: Int, d2: Int, p1: _CTPlane, p2: _CTPlane, p3: _CTPlane](
            planes: Self.Planes[p1,p2,p3]
        ):
            var p_vf = self.intersect_v[d1,d2](f_vi_corner, planes) - f_vf_corner

            var p_pf = Vec[2,SIMD[dtype,simd_width]](fill=0)
            var in_range = SIMDBool[simd_width](fill=False)
            var inclusive = SIMDBool[simd_width](fill=False)
            self.intersect_p[d1,d2,p3](f_vi, planes, p_pf, in_range, inclusive)

            var comment = String(
                "  ", p1.name, "=", d1,
                "  ", p2.name, "=", d2,
                "  p_vf=", p_vf[slice=w],
                "  in_range=", in_range[w],
                "  inclusive=", inclusive[w]
            )
            var entry = (p_pf[slice=w], comment)

            if in_range[w]:
                inside_points.append(entry)
            else:
                outside_points.append(entry)

        # collect all 12 intersection points
        classify_intersection[0,0](self.planes_xy)
        classify_intersection[0,1](self.planes_xy)
        classify_intersection[1,0](self.planes_xy)
        classify_intersection[1,1](self.planes_xy)

        classify_intersection[0,0](self.planes_xz)
        classify_intersection[0,1](self.planes_xz)
        classify_intersection[1,0](self.planes_xz)
        classify_intersection[1,1](self.planes_xz)

        classify_intersection[0,0](self.planes_yz)
        classify_intersection[0,1](self.planes_yz)
        classify_intersection[1,0](self.planes_yz)
        classify_intersection[1,1](self.planes_yz)

        var bound_pf = self.bound_pf[x_halfspace](f_vi_corner)
        var bound_pi = self.bound_pi(bound_pf)

        @parameter
        fn display_intersections(
            pts: List[Tuple[Vec[2,Scalar[dtype]],String]],
            out s: String
        ):
            s = ""
            for p in pts:
                s += "\n\tpt"
                s += String(p[0])
                s += ",  # "
                s += p[1]

        str = "Plane bound geometry:"
            + "\nf_pf=pt" + String(self.vol_to_proj(f_vi.map_scalar[dtype]())[slice=w])
            + "\nf_pf_corner=pt" + String(self.vol_to_proj(f_vf_corner)[slice=w])
            + "\naxes=["
                + "\n\tpt" + String(proj.vol_to_proj(Vec[3](x=1, y=0, z=0).map_scalar[dtype]())) + ","
                + "\n\tpt" + String(proj.vol_to_proj(Vec[3](x=0, y=1, z=0).map_scalar[dtype]())) + ","
                + "\n\tpt" + String(proj.vol_to_proj(Vec[3](x=0, y=0, z=1).map_scalar[dtype]()))
            + "\n]"
            + "\nx_halfspace=" + String(x_halfspace)
            + "\nx_len=" + String(_num_neighborhoods_in_segment[simd_width]())
            + "\nintersections_in=["
                + display_intersections(inside_points)
            + "\n]"
            + "\nintersections_out=["
                + display_intersections(outside_points)
            + "\n]"
            + "\nbound_pf=["
                + "pt" + String(bound_pf.f.min[slice=w])
                + ", pt" + String(bound_pf.f.max[slice=w])
            + "]"
            + "\nbound_pi=["
                + "pt" + String(bound_pi.f.min[slice=w])
                + ", pt" + String(bound_pi.f.max[slice=w])
            + "]"


struct _Projections[simd_width: Int, dtype: DType, *, rounding: Optional[Int] = None](
    Copyable,
    Movable
):
    var groups: List[_ProjectionGroup[dtype,simd_width,rounding=rounding]]

    fn __init__(out self, projections: List[VolumeNeighborhoodsProjection[dtype]]):

        # allocate all the groups
        var num_groups = ceildiv(len(projections), simd_width)
        self.groups = List(length=num_groups, fill=_ProjectionGroup[dtype,simd_width,rounding=rounding]())

        # populate the groups with each projection
        for p in range(len(projections)):
            ref proj = projections[p]
            var g = p // simd_width
            ref group = self.groups[g]
            var i = p % simd_width

            group.num_projections += 1
            group.proj_indices[i] = p

            # pack the factors of the vol->proj rotation matrices
            group.vol_to_proj_xfactors[slice=i] = proj.rot_proj_to_vol.vec(col=0)
            group.vol_to_proj_yfactors[slice=i] = proj.rot_proj_to_vol.vec(col=1)
            group.vol_to_proj_zfactors[slice=i] = proj.rot_proj_to_vol.vec(col=2)

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

            # init the plane info
            group.plane_x.init(i, proj)
            group.plane_y.init(i, proj)
            group.plane_z.init(i, proj)

        # init the plane pairs
        for i in range(len(self.groups)):
            ref g = self.groups[i]
            g.planes_xy.init(g.plane_x, g.plane_y, g.plane_z)
            g.planes_xz.init(g.plane_x, g.plane_z, g.plane_y)
            g.planes_yz.init(g.plane_y, g.plane_z, g.plane_x)


@fieldwise_init
struct _CTPlane(
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


struct _RTPlane[dtype: DType, simd_width: Int, plane: _CTPlane](
    Copyable,
    Movable
):
    var unit_z_component: SIMD[dtype,simd_width]

    fn __init__(out self):
        self.unit_z_component = SIMD[dtype,simd_width](0)

    fn init(mut self, i: Int, proj: VolumeNeighborhoodsProjection[dtype]):
        var unit_z_v = proj.proj_to_vol(materialize[_CTPlane.z().normal_f[dtype]()]())
        self.unit_z_component[i] = unit_z_v.inner_product(materialize[plane.normal_f[dtype]()]())


struct _RTPlanes[dtype: DType, simd_width: Int, p1: _CTPlane, p2: _CTPlane, p3: _CTPlane](
    Copyable,
    Movable
):
    var f: Vec[2,SIMD[dtype,simd_width]]

    comptime Plane = _RTPlane[dtype,simd_width,_]

    fn __init__(out self):
        self.f = Vec[2,SIMD[dtype,simd_width]](fill=0)

    fn init(
        mut self,
        _p1: Self.Plane[Self.p1],
        _p2: Self.Plane[Self.p2],
        _p3: Self.Plane[Self.p3]
    ):
        self.f = -Vec[2](
            x = _p1.unit_z_component,
            y = _p2.unit_z_component
        )/_p3.unit_z_component


@fieldwise_init
struct _PBound[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var mask: SIMDBool[simd_width]
    var f: _Bounds[2,dtype,simd_width]

    fn __init__(out self):
        self.mask = SIMDBool[simd_width](fill=False)
        self.f = _Bounds[2,dtype,simd_width]()

    fn update(
        mut self,
        in_range: SIMDBool[simd_width],
        inclusive: SIMDBool[simd_width],
        f: Vec[2,SIMD[dtype,simd_width]]
    ):
        # TODO: vectorize this?
        @parameter
        for w in range(simd_width):
            if in_range[w]:
                self.f.update[w](f, inclusive, overwrite=not self.mask[w])
            self.mask[w] = self.mask[w] or in_range[w]


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

    # TODO: vectorize this?
    fn update[w: Int](
        mut self,
        v: Vec[dim,SIMD[dtype,simd_width]],
        inclusive: SIMDBool[simd_width],
        overwrite: Bool = False
    ):
        @parameter
        for d in range(dim):
            if overwrite:
                self.min[d][w] = v[d][w]
                self.max[d][w] = v[d][w]
                self.min_inclusive[d][w] = inclusive[w]
                self.max_inclusive[d][w] = inclusive[w]
            else:
                if v[d][w] <= self.min[d][w]:
                    self.min_inclusive[d][w] = self.min_inclusive[d][w] or inclusive[w]
                if v[d][w] < self.min[d][w]:
                    self.min[d][w] = v[d][w]
                if v[d][w] >= self.max[d][w]:
                    self.max_inclusive[d][w] = self.max_inclusive[d][w] or inclusive[w]
                if v[d][w] > self.max[d][w]:
                    self.max[d][w] = v[d][w]

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
