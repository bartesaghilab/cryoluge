
from complex import ComplexSIMD

from cryoluge.io import ByteBuffer


# NOTE: coming in mojo stdlib in next release
alias ComplexScalar = ComplexSIMD[_,1]


struct ComplexImage[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    alias D1 = ComplexImage[ImageDimension.D1,_]
    alias D2 = ComplexImage[ImageDimension.D2,_]
    alias D3 = ComplexImage[ImageDimension.D3,_]
    alias VecD = VecD[_,dim]
    alias PixelType = ComplexScalar[dtype]

    fn __init__(out self, sizes: Self.VecD[UInt]):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes)
        # NOTE: This implementation uses an interleaved ordering for complex components.
        #       ie, Array-of-Structures (AoS):
        #         real_0, imag_0, real_1, imag_1, ...
        #       The other option is to use the Structure-of-Arrays (SoA) layout:
        #         real_0, real_1, ..., imag_0, imag_1, ...
        #       When thinking about SIMD operations, the different memory layouts have different performance tradeoffs.
        #       In both layouts, you need two load instructions to populate the two vector registers:
        #       One vector register for the real values, and one vector register for the imaginary values.
        #       In SoA layouts:
        #         The two loads will be adjacent in memory, making better use of (eg, L1, L2) cache.
        #         But you'll load the vector registers with interleaved data,
        #         so more vector intructions are needed
        #         to de-interleave the data into the pure real and imaginary components.
        #       In AoS layouts:
        #         The two loads will be addresses that could be far apart,
        #         which grealy reduces the effectiveness of cache.
        #         But no extra are instructions are needed to build the real and imaginary vector registers.
        #       The hypothesis is that two distant loads cost more than a few extra vector instructions,
        #       so AoS layout works best here.
        #       But, as with all things HPC, we should benchmark and be sure.
        #       For what it's worth, fftw uses the AoS layout, so that's probably good enough for us too.

    fn __init__(out self, *, sx: UInt):
        self = Self(Self.VecD[UInt](x=sx))

    fn __init__(out self, *, sx: UInt, sy: UInt):
        self = Self(Self.VecD[UInt](x=sx, y=sy))

    fn __init__(out self, *, sx: UInt, sy: UInt, sz: UInt):
        self = Self(Self.VecD[UInt](x=sx, y=sy, z=sz))

    fn rank(self) -> UInt:
        return self._buf.rank()

    fn num_pixels(self) -> UInt:
        return self._buf.num_elements()

    fn sizes(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._buf._sizes)]] Self.VecD[UInt]:
        return self._buf.sizes()

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf._buf)]]:
        return self._buf.span()

    fn __getitem__(self, i: Self.VecD[UInt], out v: Self.PixelType):
        v = self._buf[i]

    fn __getitem__(self, *, x: UInt, out v: Self.PixelType):
        v = self._buf[x=x]

    fn __getitem__(self, *, x: UInt, y: UInt, out v: Self.PixelType):
        v = self._buf[x=x, y=y]

    fn __getitem__(self, *, x: UInt, y: UInt, z: UInt, out v: Self.PixelType):
        v = self._buf[x=x, y=y, z=z]

    fn __setitem__(mut self, i: Self.VecD[UInt], v: Self.PixelType):
        self._buf[i] = v

    fn __setitem__(mut self, *, x: UInt, v: Self.PixelType):
        self._buf[x=x] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, v: Self.PixelType):
        self._buf[x=x, y=y] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, z: UInt, v: Self.PixelType):
        self._buf[x=x, y=y, z=z] = v
