
from os import abort
from sys.info import size_of


struct ImageDimension(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var rank: UInt

    # NOTE: mojo's grammar doesn't allow `1D`, `2D`, etc identifiers
    alias D1 = Self(1)
    alias D2 = Self(2)
    alias D3 = Self(3)

    fn __init__(out self, rank: UInt):
        self.rank = rank

    fn __eq__(self, rhs: Self) -> Bool:
        return self.rank == rhs.rank

    fn __ne__(self, rhs: Self) -> Bool:
        return self.rank != rhs.rank

    fn write_to[W: Writer](self, mut writer: W):
        if self.rank == Self.D1.rank:
            writer.write("D1")
        elif self.rank == Self.D2.rank:
            writer.write("D2")
        elif self.rank == Self.D3.rank:
            writer.write("D3")
        else:
            writer.write("Unknown(", self.rank, ")")

    fn __str__(self) -> String:
        return String.write(self)


fn _unrecognized_dimensionality[dim: ImageDimension, T: AnyType]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()

fn _expect_at_least_rank[dim: ImageDimension, rank: UInt]():
    constrained[
        dim.rank >= rank,
        String("Expected dimension of at least rank ", rank, " but got ", dim)
    ]()


struct DimensionalBuffer[
    dim: ImageDimension,
    T: Copyable
](
    Copyable,
    Movable
):
    var _sizes: InlineArray[UInt,dim.rank]
    var _strides: InlineArray[UInt,dim.rank]
    """the element strides, not the byte strides"""
    var _buf: ByteBuffer

    alias _elem_size = size_of[T]()

    fn __init__(out self, *, sx: UInt=1, sy: UInt=1, sz: UInt=1):
        @parameter
        if dim == ImageDimension.D1:
            debug_assert(sy == 1, "ComplexImage.D1 expects sy=1")
            debug_assert(sz == 1, "ComplexImage.D1 expects sz=1")
            self._sizes = InlineArray[UInt,dim.rank](sx)
            self._strides = InlineArray[UInt,dim.rank](1)
            self._buf = ByteBuffer(sx*Self._elem_size)
        elif dim == ImageDimension.D2:
            debug_assert(sz == 1, "ComplexImage.D2 expects sz=1")
            self._sizes = InlineArray[UInt,dim.rank](sx, sy)
            self._strides = InlineArray[UInt,dim.rank](1, sx)
            self._buf = ByteBuffer(sx*sy*Self._elem_size)
        elif dim == ImageDimension.D3:
            self._sizes = InlineArray[UInt,dim.rank](sx, sy, sz)
            self._strides = InlineArray[UInt,dim.rank](1, sx, sx*sy)
            self._buf = ByteBuffer(sx*sy*sz*Self._elem_size)
        else:
            return _unrecognized_dimensionality[dim,Self]()

    fn rank(self) -> UInt:
        return dim.rank

    fn num_elements(self) -> UInt:
        var count: UInt = 0
        @parameter
        for d in range(dim.rank):
            count *= self._sizes[d]
        return count

    fn size_x(self) -> UInt:
        _expect_at_least_rank[dim, 1]()
        return self._sizes[0]

    fn size_y(self) -> UInt:
        _expect_at_least_rank[dim, 2]()
        return self._sizes[1]

    fn size_z(self) -> UInt:
        _expect_at_least_rank[dim, 3]()
        return self._sizes[2]

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf)]]:
        return self._buf.span()

    fn _start(self) -> UnsafePointer[T]:
        return self._buf._p.bitcast[T]()

    fn _offset(self, *i: UInt) -> UInt:
        """the element offset, not the byte offset"""

        debug_assert(
            len(i) == dim.rank,
            "Image.", dim, " expects ", dim.rank, " indice(s), but got ", len(i), " indices instead"
        )

        var offset: UInt = 0

        alias d_names = InlineArray[String, 3]("x", "y", "z")
        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(coord < size, d_names[d], "=", coord, " out of range [0,", size, ")")
            offset += coord*self._strides[d]
        
        return offset

    fn __getitem__(self, *, x: UInt, out v: T):
        _expect_at_least_rank[dim, 1]()
        return (self._start() + self._offset(x))[].copy()

    fn __getitem__(self, *, x: UInt, y: UInt, out v: T):
        _expect_at_least_rank[dim, 2]()
        return (self._start() + self._offset(x, y))[].copy()

    fn __getitem__(self, *, x: UInt, y: UInt, z: UInt, out v: T):
        _expect_at_least_rank[dim, 3]()
        return (self._start() + self._offset(x, y, z))[].copy()

    fn __setitem__(self, *, x: UInt, value: T):
        _expect_at_least_rank[dim, 1]()
        (self._start() + self._offset(x))[] = value.copy()

    fn __setitem__(self, *, x: UInt, y: UInt, value: T):
        _expect_at_least_rank[dim, 2]()
        (self._start() + self._offset(x, y))[] = value.copy()

    fn __setitem__(self, *, x: UInt, y: UInt, z: UInt, value: T):
        _expect_at_least_rank[dim, 3]()
        (self._start() + self._offset(x, y, z))[] = value.copy()
