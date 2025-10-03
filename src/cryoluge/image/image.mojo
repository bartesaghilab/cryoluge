
from os import abort
from sys.info import size_of
from layout import Layout, IntTuple

from cryoluge.io import ByteBuffer


struct ImageDimension(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var size: UInt

    # NOTE: mojo's grammar doesn't allow `1D`, `2D`, etc identifiers
    alias D1 = Self(1)
    alias D2 = Self(2)
    alias D3 = Self(3)

    fn __init__(out self, size: UInt):
        self.size = size

    fn __eq__(self, rhs: Self) -> Bool:
        return self.size == rhs.size

    fn __ne__(self, rhs: Self) -> Bool:
        return self.size != rhs.size

    fn write_to[W: Writer](self, mut writer: W):
        if self.size == Self.D1.size:
            writer.write("D1")
        elif self.size == Self.D2.size:
            writer.write("D2")
        elif self.size == Self.D3.size:
            writer.write("D3")
        else:
            writer.write("Unknown(", self.size, ")")

    fn __str__(self) -> String:
        return String.write(self)


struct Image[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    var _sizes: InlineArray[UInt,dim.size]
    var _strides: InlineArray[UInt,dim.size]
    var _pixels: ByteBuffer

    alias D1 = Image[ImageDimension.D1,_]
    alias D2 = Image[ImageDimension.D2,_]
    alias D3 = Image[ImageDimension.D3,_]
    alias PixelType = Scalar[dtype]

    fn __init__(out self, *, sx: UInt=1, sy: UInt=1, sz: UInt=1):
        @parameter
        if dim == ImageDimension.D1:
            debug_assert(sy == 1, "Image.D1 expects sy=1")
            debug_assert(sz == 1, "Image.D1 expects sz=1")
            self._sizes = InlineArray[UInt,dim.size](sx)
            self._strides = InlineArray[UInt,dim.size](1)
            self._pixels = ByteBuffer(sx)
        elif dim == ImageDimension.D2:
            debug_assert(sz == 1, "Image.D2 expects sz=1")
            self._sizes = InlineArray[UInt,dim.size](sx, sy)
            self._strides = InlineArray[UInt,dim.size](1, sx)
            self._pixels = ByteBuffer(sx*sy)
        elif dim == ImageDimension.D3:
            self._sizes = InlineArray[UInt,dim.size](sx, sy, sz)
            self._strides = InlineArray[UInt,dim.size](1, sx, sx*sy)
            self._pixels = ByteBuffer(sx*sy*sz)
        else:
            return _unrecognized_dimensionality[dim,Self]()

    fn rank(self) -> UInt:
        return dim.size

    fn num_pixels(self) -> UInt:
        @parameter
        if dim == ImageDimension.D1:
            return self._sizes[0]
        elif dim == ImageDimension.D2:
            return self._sizes[0]*self._sizes[1]
        elif dim == ImageDimension.D3:
            return self._sizes[0]*self._sizes[1]*self._sizes[2]
        else:
            return _unrecognized_dimensionality[dim,UInt]()

    fn sizes(self) -> ref [self._sizes] InlineArray[UInt,dim.size]:
        return self._sizes

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._pixels)]]:
        return self._pixels.span()

    fn _offset(self, i: VariadicList[UInt]) -> UInt:

        debug_assert(
            len(i) == dim.size,
            "Image.", dim, " expects ", dim.size, " indice(s), but got ", len(i), " indices instead"
        )

        var offset: UInt = 0

        @parameter
        if dim.size >= 1:
            var x = i[0]
            var sx = self._sizes[0]
            debug_assert(x < sx, "x=", x, " out of range [0,", sx, ")")
            offset += x

        @parameter
        if dim.size >= 2:
            var y = i[1]
            var sy = self._sizes[1]
            debug_assert(y < sy, "y=", y, " out of range [0,", sy, ")")
            offset += y*self._strides[1]

        @parameter
        if dim.size >= 3:
            var z = i[2]
            var sz = self._sizes[2]
            debug_assert(z < sz, "z=", z, " out of range [0,", sz, ")")
            offset += z*self._strides[2]
        
        return offset

    fn __getitem__(self, *i: UInt) -> Self.PixelType:
        var p = self._pixels._p.bitcast[Self.PixelType]()
        p += self._offset(i)
        return p[]

    fn __setitem__(self, *i: UInt, value: Self.PixelType):
        var p = self._pixels._p.bitcast[Self.PixelType]()
        p += self._offset(i)
        p[] = value


fn _unrecognized_dimensionality[dim: ImageDimension, T: AnyType]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()
