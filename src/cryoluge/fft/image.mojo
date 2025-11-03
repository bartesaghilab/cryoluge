
from cryoluge.image import ImageDimension, VecD, ComplexImage


struct FFTImage[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: VecD[Int,dim]
    var complex: ComplexImage[dim,dtype]

    alias D1 = FFTImage[ImageDimension.D1,_]
    alias D2 = FFTImage[ImageDimension.D2,_]
    alias D3 = FFTImage[ImageDimension.D3,_]
    alias VecD = ComplexImage[dim,dtype].VecD
    alias PixelType = ComplexImage[dim,dtype].PixelType
    alias PixelVec = ComplexImage[dim,dtype].PixelVec
    alias ScalarType = ComplexImage[dim,dtype].ScalarType
    alias ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: VecD[Int,dim], *, alignment: Optional[Int] = None):
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
        fn sample(i: Self.VecD[Int]):
            dst.complex[i] = self.complex[self.coords().f2i(dst.coords().i2f(i))]

        dst.complex.iterate[sample]()
