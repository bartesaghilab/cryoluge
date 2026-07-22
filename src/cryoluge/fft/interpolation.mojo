
from math import floor
from complex import ComplexSIMD

from cryoluge.math import Dimension, Vec
from cryoluge.image import DimensionalBuffer
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
    dim: Dimension,
    dtype: DType,
    *,
    dtype_coords: DType = dtype
](Movable):
    """
    A SIMD-optimized implementation of multi-dimensional linear interpolation
    that pre-computes a cache of sampled pixel neighborhoods to maximize memory locality.
    WARNING: uses ~16x more memory than the original image data,
             which will overflow CPU caches much more quickly with increasing image sizes!
             Profiling generally shows the overall performance improvement is worth it,
             although the gains start to diminish with increasing image size.
    """
    var _sizes_real: Vec[Int,dim]
    var _samples: DimensionalBuffer[dim,Self.Pixel]

    comptime deltas = Delta[dim,dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[dtype,Self.num_samples]
    comptime EmptySamples[c: ComplexSIMD[dtype,1]] = ComplexSIMD[dtype,Self.num_samples](
        re=SIMD[dtype,Self.num_samples](c.re),
        im=SIMD[dtype,Self.num_samples](c.im)
    )
    comptime Selector = SIMD[DType.bool,Self.num_samples]

    fn __init__(
        out self,
        img: FFTImage[dim,dtype],
        out_of_range: OutOfRangeBehavior[dtype]
    ):

        self._sizes_real = img.sizes_real.copy()

        # allocate storage for all the samples
        # we'll explicitly represent the other half of the x dimension to avoid complexity due to conjugation
        var sizes = FFTCoordsFull(self._sizes_real).sizes_fourier() + 1
        # NOTE: need an extra pixel in each dimension to interpolate before fmin
        self._samples = DimensionalBuffer[dim,Self.Pixel](sizes)

        # precompute all the pixel samples
        @parameter
        fn func(i: Vec[Int,dim]):

            var f = self._i2f(i)
            var pixel = Self.Pixel(0, 0)

            @parameter
            for s in range(Self.num_samples):

                # sample the point
                var f_sample = f + materialize[Self.deltas[s].pos]()
                var v = img.find(f=f_sample)

                # handle out-of-range behavior
                if v is None:
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

    fn _coords(self) -> FFTCoordsFull[dim,origin_of(self._sizes_real)]:
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
    fn _i2f(self, i: Vec[Int,dim], out f: Vec[Int,dim]):
        """
        Maps interpolation storage coordinates into frequency coordinates.
        NOTE: This is not the same transformation as FFTCoords.i2f(),
              since the storage layouts are different.
        """

        f = Vec[Int,dim](uninitialized=True)

        @parameter
        for d in range(0, dim.rank):
            f[d] = i[d] - self._offset[d]()
        
    @always_inline
    fn _f2i[simd_width: Int](
        self,
        f: Vec[SIMDInt[simd_width],dim],
        out i: Vec[SIMDInt[simd_width],dim]
    ):
        """
        Maps frequency coordinates into the interpolation storage coordinates.
        NOTE: This is not the same transformation as FFTCoords.f2i(),
              since the storage layouts are different.
        """
        
        i = Vec[SIMDInt[simd_width],dim](uninitialized=True)

        @parameter
        for d in range(0, dim.rank):
            i[d] = f[d] + self._offset[d]()

            # if out of range, replace with -1
            var out_of_range = i[d].lt(0) or i[d].gt(self._imax[d]())
            i[d] = out_of_range.select(
                true_case = SIMDInt[simd_width](-1),
                false_case = i[d]
            )

    fn get[
        simd_width: Int,
        *,
        or_else: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
        self,
        *,
        f: Vec[SIMD[dtype_coords,simd_width],dim],
        out v: ComplexSIMD[dtype,simd_width]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[SIMDInt[simd_width],dim](uninitialized=True)
        var dists = Vec[SIMD[dtype_coords,simd_width],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            var floor = floor(f[d])
            start[d] = SIMDInt[simd_width](floor)
            dists[d] = f[d] - floor

        v = ComplexSIMD[dtype,simd_width](re=0, im=0)

        var i = self._f2i(start)
        @parameter
        for w in range(simd_width):

            # load the samples
            var samples = self._samples.get(i[slice=w].map_int())
                .or_else(Self.EmptySamples[or_else])

            # apply sample weights based on the distances
            @parameter
            for d in range(dim.rank):
                var t = SIMD[dtype,Self.num_samples](dists[d][w])
                var omt = SIMD[dtype,Self.num_samples](1 - dists[d][w])
                comptime selector = _make_selector[dim,Self.num_samples](d)
                var w = selector.select(omt, t)
                samples.re *= w
                samples.im *= w

            # the final interpolated pixel is the sum of the weighted samples
            v.re[w] = samples.re.reduce_add()
            v.im[w] = samples.im.reduce_add()


struct PrecomputedFFTInterpolationNop[
    dim: Dimension,
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
        f: Vec[SIMD[dtype_coords,simd_width],dim],
        out v: ComplexSIMD[dtype,simd_width]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[SIMDInt[simd_width],dim](uninitialized=True)
        var dists = Vec[SIMD[dtype_coords,simd_width],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
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

            # apply sample weights based on the distances
            @parameter
            for d in range(dim.rank):
                var t = SIMD[dtype,Self.num_samples](dists[d][w])
                var omt = SIMD[dtype,Self.num_samples](1 - dists[d][w])
                comptime selector = _make_selector[dim,Self.num_samples](d)
                var w = selector.select(omt, t)
                samples.re *= w
                samples.im *= w

            # the final interpolated pixel is the sum of the weighted samples
            v.re[w] = samples.re.reduce_add()
            v.im[w] = samples.im.reduce_add()


comptime _Selector[num_samples: Int] = SIMD[DType.bool,num_samples]

fn _make_selector[
    dim: Dimension,
    num_samples: Int
](d: Int, out selector: _Selector[num_samples]):
    
    comptime S = _Selector[num_samples]
    comptime t = False
    comptime omt = True
    var s0 = SIMD[DType.bool,2](omt, t)

    @parameter
    if dim.rank == 1:
        selector = rebind[S](s0)
    elif dim.rank == 2:
        if d == 0:selector = rebind[S](s0.join(s0))
        elif d == 1:
            selector = rebind[S](s0.interleave(s0))
        else:
            selector = abort[S]("d exceeds rank 2")
    elif dim.rank == 3:
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
