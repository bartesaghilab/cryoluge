
from math import floor, ceildiv
from complex import ComplexSIMD
from utils.numerics import inf

from cryoluge.math import Vec, AlignedBox, OrientedBox, complex
from cryoluge.image import DimensionalBuffer
from cryoluge.image.analysis import FrequencyLimits
from cryoluge.fft import FFTCoordsFull, Delta


comptime SIMDInt[simd_width: Int] = SIMD[DType.int,simd_width]


@fieldwise_init
struct OutOfRangeBehavior[dtype: DType](
    Movable,
    ImplicitlyCopyable
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
    comptime Selector = SIMD[DType.bool,Self.num_samples]

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


comptime _Selector[num_samples: Int] = SIMD[DType.bool,num_samples]

fn _make_selector[
    dim: Int,
    num_samples: Int
](d: Int, out selector: _Selector[num_samples]):
    
    comptime S = _Selector[num_samples]
    comptime t = False
    comptime omt = True
    var s0 = SIMD[DType.bool,2](omt, t)

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


# TEMP
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
    comptime num_neighborhoods_in_segment = simd_width - 1
    # one less neighborhood, due to needing two x voxels per neighborhood
    comptime deltas = Delta[3,dtype_coords].build()

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

            # pack all the pixels into this segment
            @parameter
            for w in range(Self.simd_width):
                # if the pixel is inside the volume, pack it
                # (otherwise, leave it out-of-range)
                var iw = i + Vec[3](x=w, y=0, z=0)
                if iw.x() < sizes_fourier.x():
                    var pixel = img.complex[i=iw]
                    segment.re[w] = pixel.re
                    segment.im[w] = pixel.im
        
            self._segments[i=s] = segment
                
        sizes_segments.iterate_over_sizes[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = sizes_fourier

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
    fn _f_in_range[x_halfspace: Int](
        self,
        *,
        f_pos: Vec[3,Int],
        out in_range: Bool
    ):
        var coords = self.coords()
        @parameter
        if x_halfspace == 1:
            in_range = f_pos.ge_all(coords.fmin()) and f_pos.le_all(coords.fmax())
        else:
            # the -x halfspace has a slightly different topology than the positive one,
            # so we need to apply different boundary conditions
            var f_neg = -f_pos - 1
            in_range = f_neg[0] >= coords.fmin[0]() and f_neg[0] <= coords.fmax[0]()
            @parameter
            for d in range(1, 3):
                in_range = in_range and f_neg[d] >= coords.fmin[d]() and f_neg[d] < coords.fmax[d]()

    @always_inline
    fn _f2i(
        self,
        *,
        f_pos: Vec[3,Int],
        out i: Vec[3,Int]
    ):
        var coords = self.coords()
        i = f_pos.copy()
        @parameter
        for d in range(1, 3):
            if f_pos[d] < 0:
                i[d] += coords.size_fourier[d]()

    @always_inline
    fn _maybe_f2i[x_halfspace: Int](
        self,
        *,
        f_pos: Vec[3,Int],
        out i: Optional[Vec[3,Int]]
    ):
        if self._f_in_range[x_halfspace](f_pos=f_pos):
            i = self._f2i(f_pos=f_pos)
        else:
            i = None

    fn _proj_bounds(
        self,
        proj: VolumeNeighborhoodsProjection[dtype],
        f_vf: Vec[3,Scalar[dtype]],
        sizes_real_proj: Vec[2,Int],
        out bounds_p: Tuple[Vec[2,Int],Vec[2,Int]]
    ):
        # define the voxel in projection space
        var f_pf = proj.vol_to_proj(f_vf)
        var voxel_p = OrientedBox(
            origin = f_pf,
            sizes = Vec[3](fill=Scalar[dtype](1)),
            orientation = proj.rot_vol_to_proj()
        )

        # find all the projection sample points within the voxel
        # by checking its axis-aligned bounding box
        # NOTE: we can get 0, 1, or 2 points, since the two grids are unit size
        var voxel_bound_p = voxel_p.bounding_box()
        var range_min_3d = voxel_bound_p.origin
            .ceil()
            .map_int()
        var range_max_3d = (voxel_bound_p.max() - voxel_p.sizes())
            .ceil()
            .map_int()

        # sample points live only on f_p = 0
        if range_min_3d.z() > 0 or range_max_3d.z() < 0:
            # out of range: return empty x,y bounds
            bounds_p = (Vec[2](fill=1), Vec[2](fill=0))
            return
        
        var range_min = range_min_3d.project[2]()
        var range_max = range_max_3d.project[2]()

        # intersect the voxel bounding box with the projection bounds
        var coords_proj = FFTCoords(sizes_real_proj)
        range_min.x() = 0
        range_min.y() = max(range_min.y(), coords_proj.fmin[1]())
        range_max = range_max.min(coords_proj.fmax())

        bounds_p = (range_min^, range_max^)

    fn _neighborhood[x_halfspace: Int](
        self,
        f_vi_pos: Vec[3,Int],
        f_vi: Vec[3,Int],
        out neighborhood: ComplexSIMD[dtype,8]
    ):

        # get the 8-voxel neighborhood image coordinates
        var i_vi_00 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=0, z=0))
        var i_vi_10 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=1, z=0))
        var i_vi_01 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=0, z=1))
        var i_vi_11 = self._maybe_f2i[x_halfspace](f_pos=f_vi_pos + Vec[3](x=0, y=1, z=1))

        # apply out-of-range behvavior
        var in_range_00 = i_vi_00 is not None
        var in_range_10 = i_vi_10 is not None
        var in_range_01 = i_vi_01 is not None
        var in_range_11 = i_vi_11 is not None
        var in_range_x = f_vi.x() < self.coords().fmax[0]()
        var all_in_range = in_range_00
            and in_range_10
            and in_range_01
            and in_range_11
            and in_range_x
        @parameter
        if out_of_range.id == OutOfRangeBehavior.Override:
            if not all_in_range:
                i_vi_00 = None
                i_vi_10 = None
                i_vi_01 = None
                i_vi_11 = None

        # read the voxel neighborhood, where possible
        var v00 = self._segment(i=i_vi_00)
        var v10 = self._segment(i=i_vi_10)
        var v01 = self._segment(i=i_vi_01)
        var v11 = self._segment(i=i_vi_11)

        # pack the neighborhood into a vector
        neighborhood = complex.pack[8](v00, v10, v01, v11)

        @parameter
        if x_halfspace == -1:

            # in the negative halfspace: re-order the voxels and conjugate
            var in_range_mask = SIMD[DType.bool,8](
                in_range_00,
                in_range_00 and in_range_x,
                in_range_10,
                in_range_10 and in_range_x,
                in_range_01,
                in_range_01 and in_range_x,
                in_range_11,
                in_range_11 and in_range_x,
            )
            neighborhood.im *= in_range_mask.select[dtype](
                true_case=-1,
                false_case=1
            )
            neighborhood.im = neighborhood.im.shuffle[7,6,5,4,3,2,1,0]()
            neighborhood.re = neighborhood.re.shuffle[7,6,5,4,3,2,1,0]()

            # f_v_neg.x = -1 (ie, f_v_pos.x = 0 in -x halfspace) doesn't have x-contiguous voxels,
            # so we need to load the missing voxels
            if f_vi_pos.x() == 0:

                # get the extra voxel image coordinates
                var f_vi_flipped = f_vi_pos*Vec[3](x=1, y=-1, z=-1)
                i_vi_00 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=0, z=0))
                i_vi_10 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=1, z=0))
                i_vi_01 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=0, z=1))
                i_vi_11 = self._maybe_f2i[1](f_pos=f_vi_flipped - Vec[3](x=0, y=1, z=1))

                # apply out-of-range behavior
                @parameter
                if out_of_range.id == OutOfRangeBehavior.Override:
                    if not all_in_range:
                        i_vi_00 = None
                        i_vi_10 = None
                        i_vi_01 = None
                        i_vi_11 = None

                # load the voxels into the neighborhood vector
                complex.splice(neighborhood, 1, self._segment(i=i_vi_11))
                complex.splice(neighborhood, 3, self._segment(i=i_vi_01))
                complex.splice(neighborhood, 5, self._segment(i=i_vi_10))
                complex.splice(neighborhood, 7, self._segment(i=i_vi_00))
                # NOTE: these voxels always load into the positive side and don't need conjugation

    fn scan[
        func: fn(proj_inf: Int, var f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]) capturing
    ](
        self,
        sizes_real_proj: Vec[2,Int],
        projections: List[VolumeNeighborhoodsProjection[dtype]],
        freq_limits: FrequencyLimits[dtype] = FrequencyLimits[dtype].none()
    ):
        var coords_proj = FFTCoords(sizes_real_proj)

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
        f_v_mini.x() = 0
        # need to push out x,y too, in both directions, to account for the inversion symmetry
        @parameter
        for d in range(1, 3):
            f_v_mini[d] = min(f_v_mini[d], -f_v_maxi[d])
            f_v_maxi[d] = max(f_v_maxi[d], -f_v_mini[d])

        # TODO: !!! implement support for SIMD segments with length > 2 !!!

        # iterate over the voxel coords that cover the projection range
        for z in range(f_v_mini.z(), f_v_maxi.z() + 1):
           for y in range(f_v_mini.y(), f_v_maxi.y() + 1):
               for x in range(0, f_v_maxi.x() + 1, self.num_neighborhoods_in_segment):

                    # TODO: can optimize by marching things along the +x,
                    #       instead of recomputing them each iteration,
                    #       like voxel bounds

                    var f_vi_pos = Vec[3](x=x, y=y, z=z)

                    # map to both positive and negative x halfspaces
                    @parameter
                    for x_halfspace in [1, -1]:
                        var f_vi = f_vi_pos.copy()
                        @parameter
                        if x_halfspace == -1:
                            f_vi = -f_vi - 1

                        var f_vf = f_vi.map_scalar[dtype]()
                        var neighborhood: Optional[ComplexSIMD[dtype,8]] = None

                        for proj in projections:

                            # compute the bounds of the voxel in projection space
                            var bounds_p = self._proj_bounds(proj, f_vf, sizes_real_proj)
                            # TODO: this is likely a bottleneck?

                            # iterate over the projection sample points in the bounding box
                            for ys in range(bounds_p[0].y(), bounds_p[1].y() + 1):
                                for xs in range(bounds_p[0].x(), bounds_p[1].x() + 1):
                                    var sf_pi = Vec[2](x=xs, y=ys)

                                    # transform back into reference volume space to compute interpolation distances
                                    var sf_vf = proj.proj_to_vol(sf_pi.map_scalar[dtype]())
                                    var dists = sf_vf - f_vf

                                    # the voxel bounding box may contain extra sample points,
                                    # so make sure the sample point actually lies inside the voxel itself
                                    # (treat the upper boundaries as exclusive)
                                    if dists.lt_any(Vec[3](fill=Scalar[dtype](0))) or dists.ge_any(Vec[3](fill=Scalar[dtype](1))):
                                        continue

                                    # apply the frequency limits
                                    var freq2 = coords_proj.freqs[dtype](f=sf_pi).len2()
                                    if not freq_limits.contains(freq2=freq2):
                                        continue

                                    # load the voxel neighborhood, if needed
                                    if neighborhood is None:
                                        neighborhood = self._neighborhood[x_halfspace](f_vi_pos, f_vi)

                                    # finally, interpolate the reference volume
                                    var sv = interpolate(dists, neighborhood.value())
                                    func(proj.id, sf_pi^, sf_vf^, sv)


@fieldwise_init
struct VolumeNeighborhoodsProjection[dtype: DType](
    Copyable,
    Movable
):
    var id: Int
    var rot_proj_to_vol: Matrix[3,3,dtype]

    fn rot_vol_to_proj(self, out rot: Matrix[3,3,dtype]):
        rot = self.rot_proj_to_vol.transposed()

    fn proj_to_vol(self, v: Vec[2,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.rot_proj_to_vol*v.lift(z=0)

    fn vol_to_proj(self, v: Vec[3,Scalar[dtype]], out result: Vec[3,Scalar[dtype]]):
        result = self.rot_proj_to_vol.mul_transpose(v)
