
from math import floor, ceildiv
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
        func: fn(proj_inf: Int, var f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]) capturing
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
        var simd_projections = _Projections[simd_width](projections)
        p.stop('projections')  # TEMP

        # iterate over the voxel coords that cover the projection range
        p.start('v-bounds')  # TEMP
        var bounds = self._voxels_bounds(coords_proj, projections)
        ref f_v_mini = bounds[0]
        ref f_v_maxi = bounds[1]
        p.stop('v-bounds')  # TEMP

        # TEMP
        ref p_rot = p.counter('rot')
        ref p_z_range = p.counter('z_range')
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

                            # TODO: optimize bounds checks when in -x halfspace
                            #       by doing f_vf.x -= n-1 before rotating into p-space?

                            p_rot.start()  # TEMP
                            var f_pf = proj_group.vol_to_proj(f_vf)
                            p_rot.stop()  # TEMP

                            # TEMP: try the new bound
                            # TODO: NEXTTIME: debug this! it's not quite working ... yet =)
                            var bounds_p = proj_group.bound_p_better[x_halfspace](f_pf)
                            var any_in_range = bounds_p.mask.reduce_or()
                            if not any_in_range:
                                continue

                            # TEMP
                            # p_z_range.start()  # TEMP
                            # var in_range_z = proj_group.in_range_z[x_halfspace](f_pf)
                            # var any_in_range = in_range_z.reduce_or()
                            # p_z_range.stop()  # TEMP
                            # if not any_in_range:
                            #     continue

                            # # compute the bounding box of the intersection of the segemnt with z_p=0
                            # p_p_bounds.start()  # TEMP
                            # var bounds_p = proj_group.bound_p[x_halfspace](f_pf, coords_proj)
                            # ref bounds_p_min = bounds_p[0]
                            # ref bounds_p_max = bounds_p[1]
                            # p_p_bounds.stop()  # TEMP

                            # for each projection in the group ...
                            for w in range(proj_group.num_projections):

                                if not bounds_p.mask[w]:
                                    continue

                                # TODO: can we get rid of this?
                                ref proj = projections[proj_group.proj_indices[w]]

                                # TEMP
                                # if debug_v:
                                #     proj_group.print_intersection_geometry[x_halfspace](w, f_pf, proj, coords_proj)

                                p_samples.start()  # TEMP

                                var segment_samples_tested = 0
                                var segment_samples_accepted = 0

                                # iterate over the projection sample points in the bounding box
                                for sy in range(bounds_p.min.y()[w], bounds_p.max.y()[w] + 1):
                                    for sx in range(bounds_p.min.x()[w], bounds_p.max.x()[w] + 1):
                                        var sf_pi = Vec[2](x=sx, y=sy).map_int()
                                        var sf_pf = sf_pi.map_scalar[dtype]()

                                        # TEMP
                                        segment_samples_tested += 1

                                        # TEMP
                                        # if debug_v:
                                        #     print("\tsampling:", sf_pi)

                                        # transform back into reference volume space
                                        p_s_rot.start()  # TEMP
                                        var sf_vf = proj.proj_to_vol(sf_pf)
                                        p_s_rot.stop()  # TEMP

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
                                        var out_of_freq = not freq_limits_proj.contains(f=sf_pf)
                                        p_s_freqs.stop()  # TEMP
                                        if out_of_freq:
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
                                samples_tested += segment_samples_tested
                                samples_accepted += segment_samples_accepted
                                # if segment_samples_accepted <= 0:
                                #     print("rejected all samples!")
                                #     proj_group.print_intersection_geometry[x_halfspace](w, f_pf, proj, coords_proj)

        # TEMP
        p.stop('scan')
        print(p)
        print("samples:",
            " tested=", samples_tested,
            ", accepted=", samples_accepted,
            " (", samples_accepted*100.0/samples_tested, "%)",
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
    fn proj_to_vol(self, v: Vec[3,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.rot_proj_to_vol*v

    @always_inline
    fn proj_to_vol(self, v: Vec[2,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.proj_to_vol(v.lift(z=0))

    @always_inline
    fn vol_to_proj[simd_width: Int](self, v: Vec[3,SIMD[dtype,simd_width]], out result: Vec[3,SIMD[dtype,simd_width]]):
        result = self.rot_proj_to_vol.mul_transpose(v)
    

comptime _VoxelNeighborhood[dtype: DType] = ComplexSIMD[dtype,8]


struct _ProjectionGroup[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var num_projections: Int
    var proj_indices: SIMDInt[simd_width]
    var segment_extents_z_neg: SIMD[dtype,simd_width]
    var segment_extents_z_pos: SIMD[dtype,simd_width]
    var vol_to_proj_xfactors: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_yfactors: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_zfactors: Vec[3,SIMD[dtype,simd_width]]
    var plane_1: _PlaneInfo[dtype,simd_width]
    var plane_2: _PlaneInfo[dtype,simd_width]
    var bound_extents_neg: Vec[2,SIMD[dtype,simd_width]]
    var bound_extents_pos: Vec[2,SIMD[dtype,simd_width]]
    # TODO: clean up unused things here
    var plane_x: _PlaneInfo[dtype,simd_width]
    var plane_y: _PlaneInfo[dtype,simd_width]
    var plane_z: _PlaneInfo[dtype,simd_width]
    var planes_xy: _PlanePairInfo[dtype,simd_width]
    var planes_xz: _PlanePairInfo[dtype,simd_width]
    var planes_yz: _PlanePairInfo[dtype,simd_width]

    fn __init__(out self):
        comptime zero_i = SIMDInt[simd_width](0)
        comptime zero_f = SIMD[dtype,simd_width](0)
        self.num_projections = 0
        self.proj_indices = zero_i
        self.segment_extents_z_neg = zero_f
        self.segment_extents_z_pos = zero_f
        self.vol_to_proj_xfactors = Vec[3](fill=zero_f)
        self.vol_to_proj_yfactors = Vec[3](fill=zero_f)
        self.vol_to_proj_zfactors = Vec[3](fill=zero_f)
        self.plane_1 = _PlaneInfo[dtype,simd_width]()
        self.plane_2 = _PlaneInfo[dtype,simd_width]()
        self.bound_extents_neg = Vec[2](fill=zero_f)
        self.bound_extents_pos = Vec[2](fill=zero_f)

        self.plane_x = _PlaneInfo[dtype,simd_width]()
        self.plane_y = _PlaneInfo[dtype,simd_width]()
        self.plane_z = _PlaneInfo[dtype,simd_width]()
        self.planes_xy = _PlanePairInfo[dtype,simd_width]()
        self.planes_xz = _PlanePairInfo[dtype,simd_width]()
        self.planes_yz = _PlanePairInfo[dtype,simd_width]()

    fn init_plane_pairs(mut self):
        self.planes_xy.set(self.plane_x, self.plane_y)
        self.planes_xz.set(self.plane_x, self.plane_z)
        self.planes_yz.set(self.plane_y, self.plane_z)

    @always_inline
    fn vol_to_proj(
        self,
        f_vf: Vec[3,Scalar[dtype]],
        out f_pf: Vec[3,SIMD[dtype,simd_width]]
    ):
        # rotate the point(s) from volume space into projection space
        var f_vf_s = f_vf.splat[simd_width]()
        f_pf = Vec[3](
            x=self.vol_to_proj_xfactors.inner_product(f_vf_s),
            y=self.vol_to_proj_yfactors.inner_product(f_vf_s),
            z=self.vol_to_proj_zfactors.inner_product(f_vf_s)
        )

    # TODO: rename me!
    # TODO: @always_inline?
    fn bound_p_better[x_halfspace: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        out bound_p: _PBound[simd_width]
    ):
        # TODO: would anything be simpler/faster if we did the intersections in volume-space?

        # TEMP
        var f_pf_adjusted = f_pf.copy()
        @parameter
        if x_halfspace == -1:
            var dx = SIMD[dtype,simd_width](self.plane_x.len - 1)
            f_pf_adjusted -= self.plane_x.normal_p*dx

        bound_p = _PBound[simd_width]()

        ref px = self.plane_x
        ref py = self.plane_y
        ref pz = self.plane_z
        ref pxy = self.planes_xy
        ref pxz = self.planes_xz
        ref pyz = self.planes_yz

        # check all 12 intersection points
        pxy.update_p_bounds[0,0](f_pf_adjusted, px, py, pz, bound_p)
        pxy.update_p_bounds[0,1](f_pf_adjusted, px, py, pz, bound_p)
        pxy.update_p_bounds[1,0](f_pf_adjusted, px, py, pz, bound_p)
        pxy.update_p_bounds[1,1](f_pf_adjusted, px, py, pz, bound_p)

        pxz.update_p_bounds[0,0](f_pf_adjusted, px, pz, py, bound_p)
        pxz.update_p_bounds[0,1](f_pf_adjusted, px, pz, py, bound_p)
        pxz.update_p_bounds[1,0](f_pf_adjusted, px, pz, py, bound_p)
        pxz.update_p_bounds[1,1](f_pf_adjusted, px, pz, py, bound_p)

        pyz.update_p_bounds[0,0](f_pf_adjusted, py, pz, px, bound_p)
        pyz.update_p_bounds[0,1](f_pf_adjusted, py, pz, px, bound_p)
        pyz.update_p_bounds[1,0](f_pf_adjusted, py, pz, px, bound_p)
        pyz.update_p_bounds[1,1](f_pf_adjusted, py, pz, px, bound_p)

    @always_inline
    fn bound_z[x_halfspace: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        out bound_z: Tuple[SIMD[dtype,simd_width],SIMD[dtype,simd_width]]
    ):
        # start with the segment extents
        var seg_z_min = self.segment_extents_z_neg
        var seg_z_max = self.segment_extents_z_pos

        @parameter
        if x_halfspace == -1:
            # in the negative half space, invert through (1,1,1)
            swap(seg_z_min, seg_z_max)
            var voxel_z_max = self.vol_to_proj_zfactors.sum()
            seg_z_min = voxel_z_max - seg_z_min
            seg_z_max = voxel_z_max - seg_z_max

        # offset the bounds by the current projection-space segment origin
        seg_z_min += f_pf.z()
        seg_z_max += f_pf.z()

        bound_z = (seg_z_min, seg_z_max)

    @always_inline
    fn in_range_z[x_halfspace: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        out in_range_z: SIMDBool[simd_width]
    ):
        var (z_min, z_max) = self.bound_z[x_halfspace](f_pf)

        # intersect the interval with z=0
        in_range_z = z_min.le(0).__and__(z_max.ge(0))

    @always_inline
    fn intersection_point[x_halfspace: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],  # point on both planes
        out p: Vec[2,SIMD[dtype,simd_width]]
    ):
        """The intersection of the two chosen planes and z_p=0."""

        var f_pf_2 = f_pf.copy()
        @parameter
        if x_halfspace == -1:
            # TODO: optimize this
            comptime dx_v = Vec[3](x=1, y=0, z=0).map_scalar[dtype]()
            var dx_p = self.vol_to_proj(materialize[dx_v]())
            f_pf_2 -= dx_p*(_num_neighborhoods_in_segment[simd_width]() - 1)

        p = Vec[2](
            x = f_pf_2.inner_product(self.plane_1.factor),
            y = f_pf_2.inner_product(self.plane_2.factor)
        )

    @always_inline
    fn bound_p[x_halfspace: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],  # point on both planes
        coords_proj: FFTCoords[2],
        out bounds: Tuple[Vec[2,SIMDInt[simd_width]],Vec[2,SIMDInt[simd_width]]]
    ):
        """The bounding box around the intersections of the both planes and their far parners at z_p=0."""
        
        # compute the intesection area bounds by adding the extents to the intersection point
        var p = self.intersection_point[x_halfspace](f_pf)
        var bound_min_f = p + self.bound_extents_neg
        var bound_max_f = p + self.bound_extents_pos
        # NOTE: don't need to adjust these extents for -x halfspace,
        #       since we already moved the intersection point to compensate

        # use the direction of the plane offsets to set the boundary conditions for the intersection area
        var far_point = self.plane_1.offset + self.plane_2.offset
        var is_pos = Vec[2](
            x = far_point.x().ge(0),
            y = far_point.y().ge(0)
        )
        # TODO: can cache this?

        # discretize the bounds and apply boundary conditions
        # when the offsets are positive, treat upper bounds as exclusive
        # when the offsets are negative, treat the lower bounds as exclusive
        var bound_min_i = Vec[2,SIMDInt[simd_width]](uninitialized=True)
        var bound_max_i = Vec[2,SIMDInt[simd_width]](uninitialized=True)
        from math import floor, ceil  # TODO: move to top
        @parameter
        for d in range(2):
            bound_min_i[d] = is_pos[d].select(
                true_case = SIMDInt[simd_width](ceil(bound_min_f[d])),
                false_case = SIMDInt[simd_width](floor(bound_min_f[d] + 1))
            )
            bound_max_i[d] = is_pos[d].select(
                true_case = SIMDInt[simd_width](ceil(bound_max_f[d] - 1)),
                false_case = SIMDInt[simd_width](floor(bound_max_f[d]))
            )

        # intersect with the projection bounds
        bound_min_i = bound_min_i.max(coords_proj.fmin_pos().map_dint().splat[simd_width]())
        bound_max_i = bound_max_i.min(coords_proj.fmax().map_dint().splat[simd_width]())

        bounds = (bound_min_i^, bound_max_i^)

    # for debugging
    fn print_intersection_geometry[x_halfspace: Int](
        self,
        i: Int,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        proj: VolumeNeighborhoodsProjection[dtype],
        coords_proj: FFTCoords[2]
    ):
        var p = self.intersection_point[x_halfspace](f_pf)[slice=i]
        var o1 = self.plane_1.offset[slice=i]
        var o2 = self.plane_2.offset[slice=i]
        var bounds_p = self.bound_p[x_halfspace](f_pf, coords_proj)
        var (z_min, z_max) = self.bound_z[x_halfspace](f_pf)
        print("Plane intersection geometry:",
            "\n\tp1=", self.plane_1.name[i], " p2=", self.plane_2.name[i],
            "\n\tf_pf=pt", f_pf[slice=i],
            "\n\taxes=[",
                "pt", proj.vol_to_proj(Vec[3](x=1, y=0, z=0).map_scalar[dtype]()),
                ", pt", proj.vol_to_proj(Vec[3](x=0, y=1, z=0).map_scalar[dtype]()),
                ", pt", proj.vol_to_proj(Vec[3](x=0, y=0, z=1).map_scalar[dtype]()),
            "]",
            "\n\tx_halfspace=", x_halfspace,
            "\n\tx_len=", _num_neighborhoods_in_segment[simd_width](),
            "\n\tintersection=pt", p,
            "\n\toffsets=[pt", o1, ", pt", o2, "]",
            "\n\textents_f=[",
                "pt", p[slice=i] + self.bound_extents_neg[slice=i],
                ", pt", p[slice=i] + self.bound_extents_pos[slice=i],
            "]",
            "\n\textents_i=[pt", bounds_p[0][slice=i], ", pt", bounds_p[1][slice=i], "]",
            "\n\tz_min=pt", Vec[3](x=p.x(), y=p.y(), z=z_min[i]),
            "\n\tz_max=pt", Vec[3](x=p.x(), y=p.y(), z=z_max[i]),
            sep=""
        )

    # for debugging
    fn render_bound_geometry[x_halfspace: Int, w: Int = 0](
        self,
        f_pf: Vec[3,Scalar[dtype]],
        proj: VolumeNeighborhoodsProjection[dtype],
        out str: String
    ):
        # TODO: move this somewhere more general!!
        var f_pf_adjusted = f_pf.copy()
        @parameter
        if x_halfspace == -1:
            # in the negative halfspace, move to the min-x corner of the segment
            var dx = Scalar[dtype](self.plane_x.len[w] - 1)
            f_pf_adjusted -= self.plane_x.normal_p[slice=w]*dx

        var f_pf_adjusted_simd = f_pf_adjusted.splat[simd_width]()

        # classify all the intersection points
        var inside_points = List[Vec[3,Scalar[dtype]]]()
        var outside_points = List[Vec[3,Scalar[dtype]]]()

        @parameter
        fn classify_intersection[d1: Int, d2: Int](
            pp: _PlanePairInfo[dtype,simd_width],
            p1: _PlaneInfo[dtype,simd_width],
            p2: _PlaneInfo[dtype,simd_width],
            p3: _PlaneInfo[dtype,simd_width]
        ):
            var i3 = pp.intersect[d1,d2](f_pf_adjusted_simd, p1, p2)
            var inside = p3.inside(i3)
            var i3_seg = (i3 + f_pf_adjusted_simd)[slice=w]
            if inside:
                inside_points.append(i3_seg^)
            else:
                outside_points.append(i3_seg^)

        ref px = self.plane_x
        ref py = self.plane_y
        ref pz = self.plane_z
        ref pxy = self.planes_xy
        ref pxz = self.planes_xz
        ref pyz = self.planes_yz

        # collect all 12 intersection points
        classify_intersection[0,0](pxy, px, py, pz)
        classify_intersection[0,1](pxy, px, py, pz)
        classify_intersection[1,0](pxy, px, py, pz)
        classify_intersection[1,1](pxy, px, py, pz)

        classify_intersection[0,0](pxz, px, pz, py)
        classify_intersection[0,1](pxz, px, pz, py)
        classify_intersection[1,0](pxz, px, pz, py)
        classify_intersection[1,1](pxz, px, pz, py)

        classify_intersection[0,0](pyz, py, pz, px)
        classify_intersection[0,1](pyz, py, pz, px)
        classify_intersection[1,0](pyz, py, pz, px)
        classify_intersection[1,1](pyz, py, pz, px)

        fn display_intersections(pts: List[Vec[3,Scalar[dtype]]], out s: String):
            s = ""
            for p in pts:
                s += "\n\tpt"
                s += String(p)
                s += ","

        str = "Plane bound geometry:"
            + "\nf_pf=pt" + String(f_pf[slice=w])
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

struct _Projections[simd_width: Int, dtype: DType](
    Copyable,
    Movable
):
    var groups: List[_ProjectionGroup[dtype,simd_width]]

    fn __init__(out self, projections: List[VolumeNeighborhoodsProjection[dtype]]):

        # allocate all the groups
        var num_groups = ceildiv(len(projections), simd_width)
        self.groups = List(length=num_groups, fill=_ProjectionGroup[dtype,simd_width]())

        # populate the groups with each projection
        for p in range(len(projections)):
            ref proj = projections[p]
            var g = p // simd_width
            ref group = self.groups[g]
            var i = p % simd_width

            group.num_projections += 1
            group.proj_indices[i] = p

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

            # pack the segment z-extents
            var seg_vec = proj.rot_proj_to_vol.vec(row=0)*(n_f - 1)
            group.segment_extents_z_neg[i] = min(voxel_extents_neg.z(), voxel_extents_neg.z() + seg_vec.z())
            group.segment_extents_z_pos[i] = max(voxel_extents_pos.z(), voxel_extents_pos.z() + seg_vec.z())

            # pack the factors of the vol->proj rotation matrices
            group.vol_to_proj_xfactors[slice=i] = proj.rot_proj_to_vol.vec(col=0)
            group.vol_to_proj_yfactors[slice=i] = proj.rot_proj_to_vol.vec(col=1)
            group.vol_to_proj_zfactors[slice=i] = proj.rot_proj_to_vol.vec(col=2)

            # compute the width of the strips in each dimension
            # ie, the minimum distance from the origin to the line formed by intersecting
            # each segment far bounding plane (not incident at origin) in volume space
            # with the z=0 plane from projection space
            var z_vec_v = proj.proj_to_vol(Vec[3](x=0, y=0, z=1).map_scalar[dtype]())
            from math import sqrt  # TODO: move to top
            var strip_widths = Vec[3](
                x = n_f*sqrt(1 + z_vec_v.x()**2/(z_vec_v.y()**2 + z_vec_v.z()**2)),
                y = sqrt(1 + z_vec_v.y()**2/(z_vec_v.x()**2 + z_vec_v.z()**2)),
                z = sqrt(1 + z_vec_v.z()**2/(z_vec_v.x()**2 + z_vec_v.y()**2))
            )

            comptime unit_x_v = Vec[3](x=1, y=0, z=0).map_scalar[dtype]()
            comptime unit_y_v = Vec[3](x=0, y=1, z=0).map_scalar[dtype]()
            comptime unit_z_v = Vec[3](x=0, y=0, z=1).map_scalar[dtype]()

            # choose the two planes we'll use for intersection tests
            from utils.numerics import isinf  # TODO: move to top
            if isinf(strip_widths.x()):
                # x doesn't intersect z_p=0, so we must use y,z
                group.plane_1.set[unit_y_v](i, "y", 1, proj)
                group.plane_2.set[unit_z_v](i, "z", 1, proj)
            elif isinf(strip_widths.y()):
                # y doesn't intersect z_p=0, so we must use x,z
                group.plane_1.set[unit_x_v](i, "x", n_i, proj)
                group.plane_2.set[unit_z_v](i, "z", 1, proj)
            elif isinf(strip_widths.z()):
                # z doesn't intersect z_p=0, so we must use x,y
                group.plane_1.set[unit_x_v](i, "x", n_i, proj)
                group.plane_2.set[unit_y_v](i, "y", 1, proj)
            else:
                # all three are possible: could choose any two
                # to optimize, pick the smaller two of the three strips
                # TODO: but don't pick two parallel strips
                ref wx = strip_widths.x()
                ref wy = strip_widths.y()
                ref wz = strip_widths.z()
                if wx < wz and wy < wz or True:  # TEMP
                    group.plane_1.set[unit_x_v](i, "x", n_i, proj)
                    group.plane_2.set[unit_y_v](i, "y", 1, proj)
                elif wx < wy and wz < wy:
                    group.plane_1.set[unit_x_v](i, "x", n_i, proj)
                    group.plane_2.set[unit_z_v](i, "z", 1, proj)
                else:
                    group.plane_1.set[unit_y_v](i, "y", 1, proj)
                    group.plane_2.set[unit_z_v](i, "z", 1, proj)

            # TEMP
            # print("strip widths=", strip_widths)
            # print("planes:", i, group.plane_1.name[i], group.plane_2.name[i])

            _PlaneInfo.mix(i, group.plane_1, group.plane_2)

            # compute the z_p=0 bound extents
            var zero_2 = Vec[2](fill=0).map_scalar[dtype]()
            var o1 = group.plane_1.offset[slice=i]
            var o2 = group.plane_2.offset[slice=i]
            group.bound_extents_neg[slice=i] = zero_2.min(o1).min(o2).min(o1 + o2)
            group.bound_extents_pos[slice=i] = zero_2.max(o1).max(o2).max(o1 + o2)
            
            # TEMP
            # print("extents=", group.bound_extents_neg[slice=i], group.bound_extents_pos[slice=i])

            # init the bound plane info
            group.plane_x.set[unit_x_v](i, "x", n_i, proj)
            group.plane_y.set[unit_y_v](i, "y", 1, proj)
            group.plane_z.set[unit_z_v](i, "z", 1, proj)
            group.planes_xy.set(group.plane_x, group.plane_y)
            group.planes_xz.set(group.plane_x, group.plane_z)
            group.planes_yz.set(group.plane_y, group.plane_z)


struct _PlaneInfo[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var name: List[String]
    var normal_p: Vec[3,SIMD[dtype,simd_width]]
    var len: SIMDInt[simd_width]
    var factor: Vec[3,SIMD[dtype,simd_width]]
    var offset: Vec[2,SIMD[dtype,simd_width]]
    # TODO: delete factor,offset?

    fn __init__(out self):
        self.name = List[String](length=simd_width, fill="")
        self.normal_p = Vec[3,SIMD[dtype,simd_width]](fill=Scalar[dtype](0))
        self.len = SIMDInt[simd_width](0)
        self.factor = Vec[3,SIMD[dtype,simd_width]](fill=Scalar[dtype](0))
        self.offset = Vec[2,SIMD[dtype,simd_width]](fill=Scalar[dtype](0))

    fn set[axis: Vec[3,Scalar[dtype]]](
        mut self,
        i: Int,
        name: String,
        len: Int,
        proj: VolumeNeighborhoodsProjection[dtype]
    ):
        self.name[i] = name
        self.normal_p[slice=i] = proj.vol_to_proj(materialize[axis]())
        self.len[i] = len

    @staticmethod
    fn mix(
        var i: Int,
        mut plane_1: Self,
        mut plane_2: Self
    ):

        # compute the plane intersection factors
        var n1_p = plane_1.normal_p[slice=i]
        var n2_p = plane_2.normal_p[slice=i]
        var d = n1_p.y()*n2_p.x() - n1_p.x()*n2_p.y()
        var f1 = (n2_p*n1_p.y() - n1_p*n2_p.y())/d
        var f2 = (n1_p*n2_p.x() - n2_p*n1_p.x())/d
        plane_1.factor[slice=i] = f1.copy()
        plane_2.factor[slice=i] = f2.copy()

        # compute the intersection corner offsets
        fn intersection_offset(
            n: Vec[3,Scalar[dtype]],
            f1: Vec[3,Scalar[dtype]],
            f2: Vec[3,Scalar[dtype]],
            out offset: Vec[2,Scalar[dtype]]
        ):
            offset = Vec[2](
                x = n.inner_product(f1),
                y = n.inner_product(f2)
            )
        plane_1.offset[slice=i] = intersection_offset(n1_p, f1, f2)*Scalar[dtype](plane_1.len[i])
        plane_2.offset[slice=i] = intersection_offset(n2_p, f1, f2)*Scalar[dtype](plane_2.len[i])

    fn inside(
        self,
        i3: Vec[3,SIMD[dtype,simd_width]],
        out inside: SIMDBool[simd_width]
    ):
        # check if the point lies in both:
        #  * the closed halfspace induced by the normal at zero
        #  * the closed halfspace induced by the negated normal at `len` units away in the normal direction
        var dot = i3.inner_product(self.normal_p)
        var len_f = SIMD[dtype](self.len)
        inside = dot.ge(0).__and__(dot.le(len_f))


struct _PlanePairInfo[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var f1: Vec[4,SIMD[dtype,simd_width]]
    var f2: Vec[4,SIMD[dtype,simd_width]]

    fn __init__(out self):
        self.f1 = Vec[4,SIMD[dtype,simd_width]](fill=0)
        self.f2 = Vec[4,SIMD[dtype,simd_width]](fill=0)

    fn set(
        mut self,
        p1: _PlaneInfo[dtype,simd_width],
        p2: _PlaneInfo[dtype,simd_width]
    ):
        ref n1 = p1.normal_p
        ref n2 = p2.normal_p

        var f1 = n2*n1.y() - n1*n2.y()
        var f2 = n1*n2.x() - n2*n1.x()
        var f3 = n2.y()*n1.z() - n1.y()*n2.z()
        var f4 = n1.x()*n2.z() - n2.x()*n1.z()

        var d = (n2.x()*n1.y() - n1.x()*n2.y())
        self.f1 = f1.lift(w=f3)/d
        self.f2 = f2.lift(w=f4)/d

    fn intersect[d1: Int, d2: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        p1: _PlaneInfo[dtype,simd_width],
        p2: _PlaneInfo[dtype,simd_width],
        out i3: Vec[3,SIMD[dtype,simd_width]]
    ):
        # translate the segment position to the origin, keep track of the offset
        var offset = -f_pf

        # compute the intersection point of the three planes
        #  * where the first plane is offset by `d1` units along the normal
        #  * the second plane is offset by `d2` units along the normal
        #  * and the third plane is z=`offset.z()`
        var n1 = p1.normal_p*SIMD[dtype,simd_width](p1.len*d1)
        var n2 = p2.normal_p*SIMD[dtype,simd_width](p2.len*d2)
        var i4 = (n1 + n2).lift(w=offset.z())
        i3 = Vec[3](
            x = i4.inner_product(self.f1),
            y = i4.inner_product(self.f2),
            z = offset.z()
        )

    fn discretize(
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        i3: Vec[3,SIMD[dtype,simd_width]],
        out bounds: _IntBounds[2,simd_width]
    ):
        # project to 2d and translate back to the segment position
        var i2 = i3.project[2]() + f_pf.project[2]()

        # discretize for the bounds
        var i2_lo = i2.ceil().map_dint()
        var i2_hi = (i2 - 1).ceil().map_dint()

        bounds = _IntBounds[2](i2_lo^, i2_hi^)

    fn update_p_bounds[d1: Int, d2: Int](
        self,
        f_pf: Vec[3,SIMD[dtype,simd_width]],
        p1: _PlaneInfo[dtype,simd_width],
        p2: _PlaneInfo[dtype,simd_width],
        p3: _PlaneInfo[dtype,simd_width],
        mut bounds: _PBound[simd_width]
    ):
        var i3 = self.intersect[d1,d2](f_pf, p1, p2)

        # check if the intersect point lies on the convex hull,
        # (ie, inside all 6 halfspaces) by checking the third plane
        var inside = p3.inside(i3)

        var i_bounds = self.discretize(f_pf, i3)

        # update the projection-space bounds
        # TODO: vectorize this
        @parameter
        for w in range(simd_width):
            if not bounds.mask[w]:
                bounds.min[slice=w] = i_bounds.min[slice=w]
                bounds.max[slice=w] = i_bounds.max[slice=w]
            else:
                bounds.min[slice=w] = bounds.min[slice=w].min(i_bounds.min[slice=w])
                bounds.max[slice=w] = bounds.max[slice=w].max(i_bounds.max[slice=w])

            # update the mask too
            bounds.mask[w] = bounds.mask[w] or inside[w]


@fieldwise_init
struct _PBound[simd_width: Int](
    Copyable,
    Movable
):
    var mask: SIMDBool[simd_width]
    var min: Vec[2,SIMDInt[simd_width]]  # inclusive
    var max: Vec[2,SIMDInt[simd_width]]  # inclusive

    fn __init__(out self):
        self.mask = SIMDBool[simd_width](fill=False)
        self.min = Vec[2,SIMDInt[simd_width]](fill=0)
        self.max = Vec[2,SIMDInt[simd_width]](fill=0)


@fieldwise_init
struct _IntBounds[dim: Int, simd_width: Int](
    Copyable,
    Movable
):
    var min: Vec[dim,SIMDInt[simd_width]]  # inclusive
    var max: Vec[dim,SIMDInt[simd_width]]  # inclusive


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
