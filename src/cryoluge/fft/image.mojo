
from math import pi, cos, sin, floor

from cryoluge.math import Dimension, Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage
from cryoluge.ctf import CTF


struct FFTImage[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: Vec[Int,dim]
    var complex: ComplexImage[dim,dtype]

    comptime D1 = FFTImage[Dimension.D1,_]
    comptime D2 = FFTImage[Dimension.D2,_]
    comptime D3 = FFTImage[Dimension.D3,_]
    comptime Vec = ComplexImage[dim,dtype].Vec
    comptime PixelType = ComplexImage[dim,dtype].PixelType
    comptime PixelVec = ComplexImage[dim,dtype].PixelVec
    comptime ScalarType = ComplexImage[dim,dtype].ScalarType
    comptime ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[dim,dtype](fft_coords.sizes_fourier(), alignment=alignment)

    fn __init__(out self, *, of: Image[dim,dtype], alignment: Optional[Int] = None):
        ref real = of
        self = Self(real.sizes(), alignment=alignment)

    fn coords(self) -> FFTCoords[dim, origin=origin_of(self.sizes_real)]:
        return FFTCoords(self.sizes_real)

    fn crop(self, *, mut to: Self):
        ref dst = to

        # make sure the destination image is smaller (or the same size) as this one
        @parameter
        for d in range(dim.rank):
            debug_assert(
                dst.sizes_real[d] <= self.sizes_real[d],
                "Crop destination real sizes ", dst.sizes_real,
                " must be smaller (or same size) as this source real sizes ", self.sizes_real
            )

        # sample into the dst image
        @parameter
        fn sample(i: Self.Vec[Int]):
            dst.complex[i] = self.complex[self.coords().f2i(dst.coords().i2f(i))]

        dst.complex.iterate[sample]()

    fn get(
        self,
        *,
        f: Self.Vec[Int],
        out v: Optional[ComplexScalar[dtype]]
    ):
        var conj = self.coords().needs_conjugation(f=f)
        var i = self.coords().maybe_f2i(f, needs_conj=conj)

        if i is None:
            v = None
            return

        v = self.complex.get(i.value())
        if v is not None and conj:
            v = v.value().conj()

    fn get(
        self: FFTImage[dim,dtype],
        *,
        f_lerp: Self.Vec[Float32]
    ) -> Optional[ComplexScalar[dtype]]:
        ref f = f_lerp

        # discretize the frequency coordinates
        @parameter
        fn func_start(v: Float32) -> Int:
            return Int(floor(v))
        var start = f.map[mapper=func_start]()

        # build the multi-dimensional delta vectors (at compile-time)
        fn build_deltas(out deltas: List[Vec[Int,dim]]):
            deltas = [
                Vec[Int,dim](fill=0)
            ]
            for d in range(dim.rank):
                for i in range(len(deltas)):
                    var delta = deltas[i].copy()
                    delta[d] = 1
                    deltas.append(delta^)

        sum = ComplexScalar[dtype](0, 0)

        @parameter
        for delta in build_deltas():
            var f_sample = start + materialize[delta]()

            # TEMP: technically, the sample might need conjugation,
            #       and the coordinate inversion might put it back in-range,
            #       but csp1 doesn't do that,
            #       so add another explicit (and unecessary) range check here to match csp1 behavior
            if not self.coords().f_in_range(f_sample):
                return None
            
            var v = self.get(f=f_sample)
            if v is None:
                return None
            var dists = (f - f_sample.map_float32()).abs()
            var weight = Scalar[dtype]((1 - dists).product())
            sum = sum + v.value()*weight

        return sum
