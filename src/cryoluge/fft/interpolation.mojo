
from math import floor, ceildiv
from complex import ComplexSIMD
from utils.numerics import inf

from cryoluge.math import Vec, AlignedBox, OrientedBox, complex, ladder
from cryoluge.image import DimensionalBuffer
from cryoluge.image.analysis import FrequencyLimits
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
    comptime num_neighborhoods_in_segment = num_neighborhoods_in_segment[simd_width]()

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

            var f = self._i2f_pos(i=i)

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

    @always_inline
    fn _i_in_range[x_halfspace: Int](
        self,
        i: Vec[3,Int],
        out in_range: Bool
    ):
        var i_min = Vec[3](fill=0)
        var i_max = self.coords().sizes_fourier()

        var i2 = i.copy()

        @parameter
        if x_halfspace == -1:
            # in the negative halfspace, the bounds are slightly (annoyingly) different
            @parameter
            for d in range(1, 3):
                i_min[d] += 1
                i2[d] = i_max[d]//2*2 - i2[d]

        in_range = i2.ge_all(i_min) and i2.lt_all(i_max)

    @always_inline
    fn _i2f_pos(
        self,
        *,
        i: Vec[3,Int],
        out f_pos: Vec[3,Int]
    ):
        # use contiguous addressing, instead of the bifurcated thing FFTCoords does
        f_pos = i + self.coords().fmin_pos()

    @always_inline
    fn _maybe_f2i[x_halfspace: Int](
        self,
        *,
        f_pos: Vec[3,Int],
        out i: Optional[Vec[3,Int]]
    ):
        # use contiguous addressing, instead of the bifurcated thing FFTCoords does
        var maybe_i = f_pos - self.coords().fmin_pos()
        if self._i_in_range[x_halfspace](maybe_i):
            i = maybe_i^
        else:
            i = None

    fn _segment_neighborhood[x_halfspace: Int](
        self,
        f_vi_pos: Vec[3,Int],
        out segment_neighborhood: _SegmentNeighborhood[dtype,simd_width]
    ):
        # get the neighborhood image coordinates
        var i_vi_00 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=0, z=0))
        var i_vi_10 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=1, z=0))
        var i_vi_01 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=0, z=1))
        var i_vi_11 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=1, z=1))

        # apply out-of-range behvavior
        var in_range_00 = i_vi_00 is not None
        var in_range_10 = i_vi_10 is not None
        var in_range_01 = i_vi_01 is not None
        var in_range_11 = i_vi_11 is not None
        var in_range_yz = in_range_10 and in_range_01
    
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not in_range_yz:
                i_vi_00 = None
                i_vi_10 = None
                i_vi_01 = None
                i_vi_11 = None

        var x_vec = materialize[ladder[simd_width]()]() + f_vi_pos.x()

        # read the segments, where possible
        segment_neighborhood = _SegmentNeighborhood[dtype,simd_width](
            s00 = self._segment(i=i_vi_00),
            s10 = self._segment(i=i_vi_10),
            s01 = self._segment(i=i_vi_01),
            s11 = self._segment(i=i_vi_11),
            in_range_x_mask = x_vec.lt(self.coords().fmax[0]())
        )

        @parameter
        if x_halfspace == -1:

            # conjugate the in-range voxels
            var in_range_x_mask = x_vec.le(self.coords().fmax[0]())
            @parameter
            if out_of_range.id == OutOfRangeBehavior.Override:
                if not in_range_yz:
                    in_range_x_mask = SIMDBool[simd_width](fill=False)

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

            # f_v_neg.x = -1 (ie, f_v_pos.x = 0 in -x halfspace) doesn't have x-contiguous voxels,
            # so we need to load the missing voxels
            if f_vi_pos.x() == 0:

                # get the extra voxel image coordinates
                var f_vi_flipped = f_vi_pos*Vec[3](x=1, y=-1, z=-1)
                var i_vi_00 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=0, z=0))
                var i_vi_10 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=1, z=0))
                var i_vi_01 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=0, z=1))
                var i_vi_11 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=1, z=1))

                # apply out-of-range behavior
                @parameter
                if out_of_range.id == OutOfRangeBehavior.Override:
                    if not in_range_yz:
                        i_vi_00 = None
                        i_vi_10 = None
                        i_vi_01 = None
                        i_vi_11 = None

                # replace the affected voxels
                complex.splice(segment_neighborhood.s00, 0, self._segment(i=i_vi_00), 0)
                complex.splice(segment_neighborhood.s10, 0, self._segment(i=i_vi_10), 0)
                complex.splice(segment_neighborhood.s01, 0, self._segment(i=i_vi_01), 0)
                complex.splice(segment_neighborhood.s11, 0, self._segment(i=i_vi_11), 0)
                # NOTE: these voxels always load into the positive side and don't need conjugation
                #       they also don't change the in-range x mask

            # re-order the voxels
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

            # and the in-order mask too
            segment_neighborhood.in_range_x_mask = segment_neighborhood.in_range_x_mask.shift_right[1]()
            segment_neighborhood.in_range_x_mask[0] = True
            segment_neighborhood.in_range_x_mask = segment_neighborhood.in_range_x_mask.reversed()

    fn scan[
        func: fn(proj_inf: Int, var f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]) capturing
    ](
        self,
        sizes_real_proj: Vec[2,Int],
        projections: List[VolumeNeighborhoodsProjection[dtype]],
        freq_limits: FrequencyLimits[dtype] = FrequencyLimits[dtype].none()
    ):
        # build the SIMD projection info
        var simd_projections = _Projections[simd_width](projections)

        var coords_proj = FFTCoords(sizes_real_proj)
        var freq_limits_proj = freq_limits.checker(sizes_real_proj)

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

        # fold the -x halfspace over the yz plane to push out the bounds
        f_v_maxi.x() = max(f_v_maxi.x(), -f_v_mini.x())
        f_v_mini.x() = max(f_v_mini.x(), 0)
        # need to push out x,y too, in both directions, to account for the inversion symmetry
        @parameter
        for d in range(1, 3):
            f_v_mini[d] = min(f_v_mini[d], -f_v_maxi[d])
            f_v_maxi[d] = max(f_v_maxi[d], -f_v_mini[d])

        # iterate over the voxel coords that cover the projection range
        for z in range(f_v_mini.z(), f_v_maxi.z() + 1):
           for y in range(f_v_mini.y(), f_v_maxi.y() + 1):
               for x in range(f_v_mini.x(), f_v_maxi.x() + 1, self.num_neighborhoods_in_segment):

                    var f_vi_pos = Vec[3](x=x, y=y, z=z)

                    # map to both positive and negative x halfspaces
                    @parameter
                    for x_halfspace in [1, -1]:
                        var f_vi = f_vi_pos.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi = -f_vi - 1
                        var f_vf = f_vi.map_scalar[dtype]()

                        var segment_neighborhood: Optional[_SegmentNeighborhood[dtype,simd_width]] = None

                        # for each group of projections ...
                        for proj_group in simd_projections.groups:

                            # check the segment z-bounds in all the projection spaces in the group
                            var in_range_z = proj_group.segment_in_range_z[x_halfspace](f_vf)
                            var any_in_range_z = in_range_z.reduce_or()
                            if not any_in_range_z:
                                continue

                            # for each projection in the group ...
                            for w in range(proj_group.num_projections):
                                ref proj = projections[proj_group.proj_indices[w]]

                                if not in_range_z[w]:
                                    continue

                                # compute projection-space bounds on each voxel in the whole segment
                                comptime x_offsets = ladder[simd_width]()*x_halfspace
                                comptime dx_vf = Vec[3](x=x_offsets, y=0, z=0).map_scalar[dtype]()
                                var f_vf_segment = f_vf.splat[simd_width]() + materialize[dx_vf]()
                                var bounds_p = proj.voxel_bounds(f_vf_segment, coords_proj)
                                ref in_range_z_segment = bounds_p[0]
                                ref bounds_p_min = bounds_p[1]
                                ref bounds_p_max = bounds_p[2]

                                # TODO: NEXTIME: try to collect sample points from all the segment voxels into a vector?
                                #                or a sequence of vectors
                                #                then calc distances and out-of-bounds together
                                #                then collapse surviving points into smaller vectors
                                #                then filter by frequency and collapse again
                                #                then send to caller

                                # for each voxel neighborhood in the segment neighborhood ...
                                # TODO: NEXTTIME: try to do vector ops here instead of iterating??
                                @parameter
                                for x_offset in range(self.num_neighborhoods_in_segment):

                                    # check z bounds first, for the easy out
                                    if not in_range_z_segment[x_offset]:
                                        continue

                                    var f_vf = f_vf + Vec[3](x=Scalar[dtype](x_offset*x_halfspace), y=0, z=0)

                                    var bounds_p_min = bounds_p_min[slice=x_offset]
                                    var bounds_p_max = bounds_p_max[slice=x_offset]

                                    # iterate over the projection sample points in the bounding box
                                    # NOTE: in 3D, with both grids being unit size, we should get 0, 1, 2 or 4 points
                                    for sy in range(bounds_p_min.y(), bounds_p_max.y() + 1):
                                        for sx in range(bounds_p_min.x(), bounds_p_max.x() + 1):
                                            var sf_pi = Vec[2](x=sx, y=sy).map_int()
                                            var sf_pf = sf_pi.map_scalar[dtype]()

                                            # TODO: dists and out-of-bounds are the new bottlneck!

                                            # transform back into reference volume space to compute interpolation distances
                                            var sf_vf = proj.proj_to_vol(sf_pf)
                                            var dists = sf_vf - f_vf

                                            # the voxel bounding box may contain extra sample points,
                                            # so make sure the sample point actually lies inside the voxel itself
                                            # (treat the upper boundaries as exclusive)
                                            var out_of_bounds = dists.lt_any(Vec[3](fill=Scalar[dtype](0))) or dists.ge_any(Vec[3](fill=Scalar[dtype](1)))
                                            if out_of_bounds:
                                                continue

                                            # apply the frequency limits
                                            var out_of_freq = not freq_limits_proj.contains(f=sf_pf)
                                            if out_of_freq:
                                                continue

                                            # load the segments, if needed
                                            if segment_neighborhood is None:
                                                segment_neighborhood = self._segment_neighborhood[x_halfspace](f_vi_pos)

                                            # finally, interpolate the reference volume
                                            var voxel_neighborhood = segment_neighborhood.value().voxel_neighborhood[
                                                _map_offset(x_halfspace,x_offset, Self.num_neighborhoods_in_segment),
                                                Self.out_of_range
                                            ]()
                                            var sv = interpolate(dists, voxel_neighborhood)
                                            func(proj.id, sf_pi^, sf_vf^, sv)


fn num_neighborhoods_in_segment[simd_width: Int]() -> Int:
    return simd_width - 1
    # one less neighborhood, due to needing two x voxels per neighborhood


struct VolumeNeighborhoodsProjection[dtype: DType](
    Copyable,
    Movable
):
    var id: Int
    var rot_proj_to_vol: Matrix[3,3,dtype]
    var _voxel_extents_neg: Vec[3,Scalar[dtype]]
    var _voxel_extents_pos: Vec[3,Scalar[dtype]]

    fn __init__(
        out self,
        id: Int,
        rot_proj_to_vol: Matrix[3,3,dtype]
    ):
        self.id = id
        self.rot_proj_to_vol = rot_proj_to_vol.copy()

        # compute the bounding volume extents of the unit voxel,
        # in projection space, relative to the voxel origin
        var voxel_bound = OrientedBox(
            origin = Vec[3](fill=Scalar[dtype](0)),
            sizes = Vec[3](fill=Scalar[dtype](1)),
            orientation = rot_proj_to_vol.transposed()
        ).bounding_box()
        self._voxel_extents_neg = voxel_bound.origin.copy()
        self._voxel_extents_pos = voxel_bound.max()

    @always_inline
    fn proj_to_vol(self, v: Vec[2,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.rot_proj_to_vol*v.lift(z=0)

    @always_inline
    fn vol_to_proj[simd_width: Int](self, v: Vec[3,SIMD[dtype,simd_width]], out result: Vec[3,SIMD[dtype,simd_width]]):
        result = self.rot_proj_to_vol.mul_transpose(v)
    
    @always_inline
    fn voxel_bounds[simd_width: Int](
        self,
        f_vf: Vec[3,SIMD[dtype,simd_width]],
        coords_proj: FFTCoords[2],
        out bounds: Tuple[SIMDBool[simd_width],Vec[2,SIMDInt[simd_width]],Vec[2,SIMDInt[simd_width]]]
    ):
        # rotate the voxel origin into projection space
        var f_pf = self.vol_to_proj(f_vf)

        # build the bounding box extents and discretize
        var range_min_3d = (f_pf + self._voxel_extents_neg.splat[simd_width]()).ceil().map_dint()
        var range_max_3d = (f_pf + self._voxel_extents_pos.splat[simd_width]()).map_dint()

        # build the z-in-range mask
        z_mask = range_min_3d.z().le(0).__and__(range_max_3d.z().ge(0))

        # intersect with the projection bounds
        var range_min_2d = range_min_3d.project[2]().max(coords_proj.fmin_pos().map_dint().splat[simd_width]())
        var range_max_2d = range_max_3d.project[2]().min(coords_proj.fmax().map_dint().splat[simd_width]())

        bounds = (z_mask, range_min_2d^, range_max_2d^)


comptime _VoxelNeighborhood[dtype: DType] = ComplexSIMD[dtype,8]


struct _ProjectionGroup[dtype: DType, simd_width: Int](
    Copyable,
    Movable
):
    var num_projections: Int
    var proj_indices: SIMDInt[simd_width]
    var voxel_extents_neg: Vec[3,SIMD[dtype,simd_width]]
    var voxel_extents_pos: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_zvec: Vec[3,SIMD[dtype,simd_width]]
    var vol_to_proj_xz: SIMD[dtype,simd_width]

    fn __init__(out self):
        self.num_projections = 0
        self.proj_indices = SIMDInt[simd_width](0)
        self.voxel_extents_neg = Vec[3](fill=SIMD[dtype,simd_width](0))
        self.voxel_extents_pos = Vec[3](fill=SIMD[dtype,simd_width](0))
        self.vol_to_proj_zvec = Vec[3](fill=SIMD[dtype,simd_width](0))
        self.vol_to_proj_xz = 0

    @always_inline
    fn segment_in_range_z[x_halfspace: Int](
        self,
        f_vf: Vec[3,Scalar[dtype]],
        out in_range: SIMDBool[simd_width]
    ):
        # rotate the voxel origin into projection space, but only the z-coord
        # by using the z-row of the vol->proj rotation matrix
        var fz_pf = self.vol_to_proj_zvec.inner_product(f_vf.splat[simd_width]())

        # get the change in z-coord along the segment
        # by using the z-coord of the unit x vector in the vol->proj rotation matrix
        var dz = self.vol_to_proj_xz*x_halfspace*num_neighborhoods_in_segment[simd_width]()

        # the z-bounds of the whole segment must contain 0 to be in-range
        var z_neg = -self.voxel_extents_neg.z()
        var z_pos = -self.voxel_extents_pos.z()
        in_range = (fz_pf.le(z_neg).__or__((fz_pf + dz).le(z_neg)))
            .__and__(fz_pf.ge(z_pos).__or__((fz_pf + dz).ge(z_pos)))
        # NOTE: and/or operators don't do what you'd expect on SIMD bools,
        #       so use the __and__/__or__ functions explicitly


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
            
            # pack the voxel extents
            group.voxel_extents_neg[slice=i] = proj._voxel_extents_neg.copy()
            group.voxel_extents_pos[slice=i] = proj._voxel_extents_pos.copy()

            # pack the z vectors of the vol->proj rotation matrices
            group.vol_to_proj_zvec[slice=i] = proj.rot_proj_to_vol.vec(col=2)

            # pack the z-coordinates of the x vectors of the vol->proj rotation matrices
            group.vol_to_proj_xz[i] = proj.rot_proj_to_vol[0,2]


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
    comptime num_neighborhoods_in_segment = num_neighborhoods_in_segment[simd_width]()

    fn voxel_neighborhood[
        x_offset: Int,
        out_of_range: OutOfRangeBehavior[dtype]
    ](self, out voxel_neighborhood: _VoxelNeighborhood[dtype]):

        # apply out-of-range behavior
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not self.in_range_x_mask[x_offset]:
                voxel_neighborhood = complex.pack[8](
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value,
                    out_of_range.value
                )
                return

        voxel_neighborhood = complex.pack[8](
            complex.slice[x_offset,2](self.s00),
            complex.slice[x_offset,2](self.s10),
            complex.slice[x_offset,2](self.s01),
            complex.slice[x_offset,2](self.s11)
        )


fn _map_offset(x_halfspace: Int, x_offset: Int, num_neighborhoods_in_segment: Int) -> Int:
    if x_halfspace == 1:
        return x_offset
    else:
        return num_neighborhoods_in_segment - x_offset - 1
