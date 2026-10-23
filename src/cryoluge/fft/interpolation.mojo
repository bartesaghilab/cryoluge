
from std.math import floor, ceil, ceildiv
from std.complex import ComplexScalar, ComplexSIMD
from std.utils.numerics import inf, nan, isinf, isnan
from std.os import abort

from cryoluge.math import Vec, AlignedBox, OrientedBox, complex, ladder, round
from cryoluge.math.dimension import unimplemented_dimension
from cryoluge.image import DimensionalBuffer
from cryoluge.image.analysis import FrequencyLimits, FrequencyLimitsChecker
from cryoluge.fft.coords import FFTCoordsFull
from cryoluge.fft.image import FFTImage, Delta


@fieldwise_init
struct OutOfRangeBehavior[dtype: DType](
    Movable,
    ImplicitlyCopyable,
    Writable
):
    var id: Int
    var value: ComplexScalar[Self.dtype]

    comptime Interpolate: Int = 1
    comptime Override: Int = 2

    @staticmethod
    def interpolate(v: ComplexScalar[Self.dtype], out s: Self):
        s = Self(Self.Interpolate, v)

    @staticmethod
    def override(v: ComplexScalar[Self.dtype], out s: Self):
        s = Self(Self.Override, v)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("OutOfRangeBehavior[")
        if self.id == Self.Interpolate:
            writer.write("interpolate")
        elif self.id == Self.Override:
            writer.write("override")
        else:
            writer.write("(unknown)")
        writer.write(", v=", self.value, "]")


struct PrecomputedFFTInterpolationFull[
    dim: Int,
    dtype: DType,
    out_of_range: OutOfRangeBehavior[dtype],
    *,
    dtype_coords: DType = dtype,
](Movable)
    where dim > 0:
    """
    A SIMD-optimized implementation of multi-dimensional linear interpolation
    that pre-computes a cache of sampled pixel neighborhoods to maximize memory locality.
    WARNING: uses ~16x more memory than the original image data,
             which will overflow CPU caches much more quickly with increasing image sizes!
             Profiling generally shows the overall performance improvement is worth it,
             although the gains start to diminish with increasing image size.
    """
    var _sizes_real: Vec[Self.dim, .int]
    var _samples: DimensionalBuffer[Self.dim,Self.Pixel]

    comptime deltas = Delta[Self.dim,Self.dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[Self.dtype,Self.num_samples]
    comptime empty_samples = ComplexSIMD[Self.dtype,Self.num_samples](
        re=Self.out_of_range.value.re,
        im=Self.out_of_range.value.im
    )
    comptime Selector = _Selector[Self.num_samples]

    def __init__(
        out self,
        img: FFTImage[Self.dim,Self.dtype]
    ):

        self._sizes_real = img.sizes_real.copy()

        # allocate storage for all the samples
        # we'll explicitly represent the other half of the x dimension to avoid complexity due to conjugation
        var sizes = FFTCoordsFull(self._sizes_real).sizes_fourier() + 1
        # NOTE: need an extra pixel in each dimension to interpolate before fmin
        self._samples = DimensionalBuffer[Self.dim,Self.Pixel](sizes)

        # precompute all the pixel samples
        def func(i: Vec[Self.dim, .int]) {mut self, img}:

            var f = self._i2f(i)
            var pixel = Self.Pixel(0, 0)

            comptime for s in range(Self.num_samples):

                # sample the point
                var f_sample = f + materialize[Self.deltas[s].pos]()
                var v = img.find(f=f_sample)

                # handle out-of-range behavior
                if v is None:
                    comptime if Self.out_of_range.id == OutOfRangeBehavior.Interpolate:
                        # interpolate with the out-of-range value
                        pixel.re[s] = Self.out_of_range.value.re
                        pixel.im[s] = Self.out_of_range.value.im
                    elif Self.out_of_range.id == OutOfRangeBehavior.Override:
                        # override the whole pixel with the out-of-range value
                        pixel.re = SIMD[Self.dtype,Self.num_samples](Self.out_of_range.value.re)
                        pixel.im = SIMD[Self.dtype,Self.num_samples](Self.out_of_range.value.im)
                        break
                else:
                    # otherwise, just interpolate with the sampled value like normal
                    pixel.re[s] = v.value().re
                    pixel.im[s] = v.value().im

            self._samples[i=i] = pixel
                
        self._samples.elements().iterate(func)

    def _coords(self) -> FFTCoordsFull[Self.dim]:
        return FFTCoordsFull(self._sizes_real)

    @inline(.always)
    def _offset[d: Int](self, out offset: Int):
        offset = (self._sizes_real[d] + 2) >> 1

    @inline(.always)
    def _imax[d: Int](self, out imax: Int):
        imax = self._sizes_real[d]
        comptime if d == 0:
            imax |= 0b1

    @inline(.always)
    def _i2f(self, i: Vec[Self.dim, .int], out f: Vec[Self.dim, .int]):
        """
        Maps interpolation storage coordinates into frequency coordinates.
        NOTE: This is not the same transformation as FFTCoords.i2f(),
              since the storage layouts are different.
        """

        f = Vec[Self.dim, .int](uninitialized=True)

        comptime for d in range(0, Self.dim):
            f[d] = i[d] - self._offset[d]()
        
    @inline(.always)
    def _f2i[slen: SIMDLength](
        self,
        f: Vec[Self.dim,.int,slen],
        out i: Vec[Self.dim,.int,slen]
    ):
        """
        Maps frequency coordinates into the interpolation storage coordinates.
        NOTE: This is not the same transformation as FFTCoords.f2i(),
              since the storage layouts are different.
        """
        
        i = Vec[Self.dim,.int,slen](uninitialized=True)

        comptime for d in range(0, Self.dim):
            i[d] = f[d] + self._offset[d]()

            # if out of range, replace with -1
            var out_of_range = i[d].lt(0) or i[d].gt(self._imax[d]())
            i[d] = out_of_range.select(
                true_case = SIMD[.int,slen](-1),
                false_case = i[d]
            )

    def _start_dists[slen: SIMDLength](
        self,
        f: Vec[Self.dim,Self.dtype_coords,slen],
        out result: Tuple[
            Vec[Self.dim,.int,slen],
            Vec[Self.dim,Self.dtype_coords,slen]
        ]
    ):

        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[Self.dim,.int,slen](uninitialized=True)
        var dists = Vec[Self.dim,Self.dtype_coords,slen](uninitialized=True)
        comptime for d in range(Self.dim):
            var floor = floor(f[d])
            start[d] = SIMD[.int,slen](floor)
            dists[d] = f[d] - floor

        result = (start^, dists^)

    def _neighborhood(
        self,
        *,
        i: Vec[Self.dim,.int],
        out v: ComplexSIMD[Self.dtype,Self.num_samples]
    ):
        v = self._samples.get(i)
            .or_else(Self.empty_samples)

    def get[slen: SIMDLength](
        self,
        *,
        f: Vec[Self.dim,Self.dtype_coords,slen],
        out v: ComplexSIMD[Self.dtype,slen]
    ):
        var result = self._start_dists(f)
        ref start = result[0]
        ref dists = result[1]

        var i = self._f2i(start)

        v = ComplexSIMD[Self.dtype,slen](re=0, im=0)

        comptime for w in range(slen):
            comptime assert w >= 0 and w < Self.dim  # not provided by `range` I guess?

            var neighborhood = self._neighborhood(i=i[slice=w].map_dtype[.int]())
            var vw = interpolate(dists[slice=w], neighborhood)
            v.re[w] = vw.re
            v.im[w] = vw.im


def interpolate[
    dim: Int,
    dtype: DType,
    dtype_coords: DType,
    num_samples: Int
](
    dists: Vec[dim,dtype_coords],
    var samples: ComplexSIMD[dtype,num_samples],
    out v: ComplexScalar[dtype]
):

    v = ComplexScalar[dtype](re=0, im=0)

    # apply sample weights based on the distances
    comptime for d in range(dim):
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
](Movable) where dim > 0:
    """
    A no-op implementation of the FFT interpolation, for testing,
    to see how well (or poorly) doing the interpolation with incoherent memory accesses really is.
    NOTE: It's very slow compared to the real pre-computed interpolation,
          but it doesn't suffer from the 16x memory requirements
          and resultant scaling issues due to limited CPU cache sizes.
    """
    var _img: FFTImage[Self.dim,Self.dtype]

    comptime deltas = Delta[Self.dim,Self.dtype_coords].build()
    comptime num_samples = len(Self.deltas)
    comptime Pixel = ComplexSIMD[Self.dtype,Self.num_samples]
    comptime EmptySamples[c: ComplexSIMD[Self.dtype,1]] = ComplexSIMD[Self.dtype,Self.num_samples](
        re=SIMD[Self.dtype,Self.num_samples](c.re),
        im=SIMD[Self.dtype,Self.num_samples](c.im)
    )

    def __init__(
        out self,
        img: FFTImage[Self.dim,Self.dtype]
    ):
        self._img = img.copy()

    def get[slen: SIMDLength](
        self,
        *,
        f: Vec[Self.dim,Self.dtype_coords,slen],
        out v: ComplexSIMD[Self.dtype,slen]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[Self.dim,.int,slen](uninitialized=True)
        var dists = Vec[Self.dim,Self.dtype_coords,slen](uninitialized=True)
        comptime for d in range(Self.dim):
            var floor = floor(f[d])
            start[d] = SIMD[.int,slen](floor)
            dists[d] = f[d] - floor

        v = ComplexSIMD[Self.dtype,slen](re=0, im=0)

        comptime for w in range(slen):
            comptime assert w >= 0 and w < Self.dim  # not provided by `range` I guess?

            # load the samples
            # NOTE: this part just loads all 2,4, or 8 pixels independently,
            #       hoping that limited locality in the x-dimension will give somewhat good cache performance
            var samples = Self.Pixel(0, 0)
            comptime for s in range(Self.num_samples):
                var f_sample = start[slice=w].map_dtype[.int]() + materialize[Self.deltas[s].pos]()
                var v = self._img.find(f=f_sample)

                # handle out-of-range behavior
                if v is None:
                    comptime if Self.out_of_range.id == OutOfRangeBehavior.Interpolate:
                        # interpolate with the out-of-range value
                        samples.re[s] = Self.out_of_range.value.re
                        samples.im[s] = Self.out_of_range.value.im
                    elif Self.out_of_range.id == OutOfRangeBehavior.Override:
                        # override the whole pixel with the out-of-range value
                        samples.re = SIMD[Self.dtype,Self.num_samples](Self.out_of_range.value.re)
                        samples.im = SIMD[Self.dtype,Self.num_samples](Self.out_of_range.value.im)
                        break
                else:
                    # otherwise, just interpolate with the sampled value like normal
                    samples.re[s] = v.value().re
                    samples.im[s] = v.value().im

            var vw = interpolate(dists[slice=w], samples)
            v.re[w] = vw.re
            v.im[w] = vw.im


comptime _Selector[num_samples: Int] = SIMD[.bool,num_samples]


def _make_selector[
    dim: Int,
    num_samples: Int
](d: Int, out selector: _Selector[num_samples]):
    
    comptime S = _Selector[num_samples]
    comptime t = False
    comptime omt = True
    var s0 = SIMD[.bool,2](omt, t)

    comptime if dim == 1:
        selector = rebind[S](s0)
    elif dim == 2:
        if d == 0:
            selector = rebind[S](s0.join(s0))
        elif d == 1:
            selector = rebind[S](s0.interleave(s0))
        else:
            abort("d exceeds rank 2")
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
            abort("d exceeds rank 3")
    else:
        unimplemented_dimension[dim]()


comptime PrecomputedFFTInterpolation = PrecomputedFFTInterpolationNop
# NOTE: this is useful for switching downstream apps to use different implementations during benchmarking
