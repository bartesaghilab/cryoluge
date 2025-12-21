
from math import floor
from complex import ComplexSIMD

from cryoluge.math import Dimension, Vec
from cryoluge.image import DimensionalBuffer
from cryoluge.fft import FFTCoordsFull, Delta


struct PrecomputedFFTInterpolation[
    dim: Dimension,
    dtype: DType,
    *,
    dtype_coords: DType = DType.float32
](Movable):
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

    fn __init__(out self, img: FFTImage[dim,dtype]):

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
                var f_sample = f + materialize[Self.deltas[s].pos]()
                var v = img.get[or_else=ComplexScalar[dtype](0, 0)](f=f_sample)
                pixel.re[s] = v.re
                pixel.im[s] = v.im

            self._samples[i=i] = pixel
                
        self._samples.iterate[func]()

        _ = coords  # TEMP: extend lifetimes to work around compiler bug

    fn _coords(self) -> FFTCoordsFull[dim,origin_of(self._sizes_real)]:
        return FFTCoordsFull(self._sizes_real)

    @always_inline
    fn _f2i(self, f: Vec[Int,dim], out i: Vec[Int,dim]):
        
        i = Vec[Int,dim](uninitialized=True)

        @parameter
        for d in range(dim.rank):
            if f[d] < 0:
                i[d] = f[d] + self._coords().size_fourier[d]() + 1
            else:
                i[d] = f[d]
    
    @always_inline
    fn _i2f(self, i: Vec[Int,dim], out f: Vec[Int,dim]):

        f = Vec[Int,dim](uninitialized=True)

        @parameter
        for d in range(0, dim.rank):
            if i[d] >= self._coords()._pivot[d]():
                f[d] = i[d] - self._coords().size_fourier[d]() - 1
            else:
                f[d] = i[d]

    fn get[
        *,
        or_else: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
        self,
        *,
        f: Vec[Scalar[dtype_coords],dim],
        out v: ComplexScalar[dtype]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[Int,dim](uninitialized=True)
        var dists = Vec[Scalar[dtype_coords],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            var floor = floor(f[d])
            start[d] = Int(floor)
            dists[d] = f[d] - floor

        # load the samples
        var i = self._f2i(start)
        var samples = self._samples.get(i)
           .or_else(Self.EmptySamples[or_else])

        # apply sample weights based on the distances
        @parameter
        for d in range(dim.rank):
            var t = SIMD[dtype,Self.num_samples](dists[d])
            var omt = SIMD[dtype,Self.num_samples](1 - dists[d])
            comptime selector = Self.make_selector(d)
            var w = selector.select(omt, t)
            samples.re *= w
            samples.im *= w

        # the final interpolated pixel is the sum of the weighted samples
        v = ComplexScalar[dtype](
            re = samples.re.reduce_add(),
            im = samples.im.reduce_add()
        )

    @staticmethod
    fn make_selector(d: Int, out selector: Self.Selector):
                    
        comptime t = False
        comptime omt = True
        var s0 = SIMD[DType.bool,2](omt, t)

        @parameter
        if dim.rank == 1:
            selector = rebind[Self.Selector](s0)
        elif dim.rank == 2:
            if d == 0:
                selector = rebind[Self.Selector](s0.join(s0))
            elif d == 1:
                selector = rebind[Self.Selector](s0.interleave(s0))
            else:
                selector = abort[Self.Selector]("d exceeds rank 2")
        elif dim.rank == 3:
            if d == 0:
                var s1 = s0.join(s0)
                selector = rebind[Self.Selector](s1.join(s1))
            elif d == 1:
                var s1 = s0.interleave(s0)
                selector = rebind[Self.Selector](s1.join(s1))
            elif d == 2:
                var s1 = s0.interleave(s0)
                selector = rebind[Self.Selector](s1.interleave(s1))
            else:
                selector = abort[Self.Selector]("d exceeds rank 3")
        else:
            constrained[False, String("unrecognized dimension: ", dim)]()
            selector = abort[Self.Selector]()
