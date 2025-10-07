
from cryoluge.io import ByteBuffer


struct Image[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    alias D1 = Image[ImageDimension.D1,_]
    alias D2 = Image[ImageDimension.D2,_]
    alias D3 = Image[ImageDimension.D3,_]
    alias VecD = VecD[_,dim]
    alias PixelType = Scalar[dtype]

    fn __init__(out self, sizes: Self.VecD[UInt]):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes)

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
