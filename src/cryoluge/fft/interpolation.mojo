
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
]:
    var _sizes_real: Vec[Int,dim]
    var _samples: DimensionalBuffer[dim,Self.Pixel]

    comptime deltas = Delta[dim,dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[dtype,Self.num_samples]

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
        # TODO: can we vectorize this?

        # load the samples
        var i = self._f2i(start)
        var samples = self._samples.get(i)
        if samples is None:
            return materialize[or_else]()
        
        # compute the sample weights based on the distances
        var weights = SIMD[dtype,Self.num_samples](0)
        @parameter
        for s in range(Self.num_samples):
            var dir = materialize[Self.deltas[s].dir]()
            var pos = materialize[Self.deltas[s].pos_f]()
            weights[s] = Scalar[dtype]((dists*dir + pos).product())
        # TODO: can we vectorize this?

        # compute the weighted sum
        v = ComplexScalar[dtype](
            re = (samples.value().re * weights).reduce_add(),
            im = (samples.value().im * weights).reduce_add()
        )
