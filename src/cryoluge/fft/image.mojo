
from math import pi, cos, sin

from cryoluge.math import Dimension, Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage


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

    alias D1 = FFTImage[Dimension.D1,_]
    alias D2 = FFTImage[Dimension.D2,_]
    alias D3 = FFTImage[Dimension.D3,_]
    alias Vec = ComplexImage[dim,dtype].Vec
    alias PixelType = ComplexImage[dim,dtype].PixelType
    alias PixelVec = ComplexImage[dim,dtype].PixelVec
    alias ScalarType = ComplexImage[dim,dtype].ScalarType
    alias ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[dim,dtype](fft_coords.sizes_fourier(), alignment=alignment)

    fn __init__(out self, *, of: Image[dim,dtype], alignment: Optional[Int] = None):
        ref real = of
        self = Self(real.sizes(), alignment=alignment)

    fn coords(self) -> FFTCoords[dim, origin=__origin_of(self.sizes_real)]:
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

    fn phase_shift(mut self, shift: Self.Vec[Scalar[dtype]]):
        
        @parameter
        fn func(i: Self.Vec[Int]):
            var freq = self.coords().i2f(i).map_scalar[dtype]()
            var sizes_real = self.coords().sizes_real().map_scalar[dtype]()
            var phase = 0 - (freq*shift*2*pi/sizes_real).sum()
            self.complex[i=i] *= ComplexScalar[dtype](re=cos(phase), im=sin(phase))

        self.complex.iterate[func]()
