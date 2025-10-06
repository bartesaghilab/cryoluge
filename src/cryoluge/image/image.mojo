
from cryoluge.io import ByteBuffer

from .dimension import _unrecognized_dimensionality


struct Image[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    alias D1 = Image[ImageDimension.D1,_]
    alias D2 = Image[ImageDimension.D2,_]
    alias D3 = Image[ImageDimension.D3,_]
    alias PixelType = Scalar[dtype]

    fn __init__(out self, *, sx: UInt=1, sy: UInt=1, sz: UInt=1):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sx=sx, sy=sy, sz=sz)

    fn rank(self) -> UInt:
        return self._buf.rank()

    fn num_pixels(self) -> UInt:
        return self._buf.num_elements()

    fn size_x(self) -> UInt:
        return self._buf.size_x()

    fn size_y(self) -> UInt:
        return self._buf.size_y()

    fn size_z(self) -> UInt:
        return self._buf.size_z()

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf._buf)]]:
        return self._buf.span()

    fn __getitem__(self, *, x: UInt, out v: Self.PixelType):
        v = self._buf[x=x]

    fn __getitem__(self, *, x: UInt, y: UInt, out v: Self.PixelType):
        v = self._buf[x=x, y=y]

    fn __getitem__(self, *, x: UInt, y: UInt, z: UInt, out v: Self.PixelType):
        v = self._buf[x=x, y=y, z=z]

    fn __setitem__(self, *, x: UInt, value: Self.PixelType):
        self._buf[x=x] = value

    fn __setitem__(self, *, x: UInt, y: UInt, value: Self.PixelType):
        self._buf[x=x, y=y] = value

    fn __setitem__(self, *, x: UInt, y: UInt, z: UInt, value: Self.PixelType):
        self._buf[x=x, y=y, z=z] = value
