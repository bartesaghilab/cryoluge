
from os import abort
from sys.info import size_of


@fieldwise_init
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

    fn __eq__(self, rhs: Self) -> Bool:
        return self.rank == rhs.rank

    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.D1:
            writer.write("D1")
        elif self == Self.D2:
            writer.write("D2")
        elif self == Self.D3:
            writer.write("D3")
        else:
            writer.write("Unknown(", self.rank, ")")

    fn __str__(self) -> String:
        return String.write(self)


fn unrecognized_dimensionality[dim: ImageDimension, T: AnyType]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()


fn expect_at_least_rank[dim: ImageDimension, rank: UInt]():
    constrained[
        dim.rank >= rank,
        String("Expected dimension of at least rank ", rank, " but got ", dim)
    ]()


fn expect_num_arguments[dim: ImageDimension, count: UInt]():
    constrained[
        dim.rank == count,
        String(dim, " expects ", dim.rank, " argument(s), but got ", count, " instead")
    ]()


struct DimensionalBuffer[
    dim: ImageDimension,
    T: Copyable
](
    Copyable,
    Movable
):
    var _sizes: Self.VecD[UInt]
    var _strides: Self.VecD[UInt]
    """the element strides, not the byte strides"""
    var _buf: ByteBuffer

    alias VecD = VecD[_,dim]
    alias _elem_size = size_of[T]()

    fn __init__(out self, sizes: Self.VecD[UInt]):
        self._sizes = sizes.copy()
        @parameter
        if dim == ImageDimension.D1:
            var sx = self._sizes.x()
            self._strides = Self.VecD[UInt](x=1)
            self._buf = ByteBuffer(sx*Self._elem_size)
        elif dim == ImageDimension.D2:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            self._strides = Self.VecD[UInt](x=1, y=sx)
            self._buf = ByteBuffer(sx*sy*Self._elem_size)
        elif dim == ImageDimension.D3:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            var sz = self._sizes.z()
            self._strides = Self.VecD[UInt](x=1, y=sx, z=sx*sy)
            self._buf = ByteBuffer(sx*sy*sz*Self._elem_size)
        else:
            return unrecognized_dimensionality[dim,Self]()

    fn rank(self) -> UInt:
        return dim.rank

    fn num_elements(self) -> UInt:
        var count: UInt = 1
        @parameter
        for d in range(dim.rank):
            count *= self._sizes[d]
        return count

    fn sizes(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._sizes)]] Self.VecD[UInt]:
        return self._sizes

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf)]]:
        return self._buf.span()

    fn _start(self) -> UnsafePointer[T]:
        return self._buf._p.bitcast[T]()

    fn _offset(self, i: Self.VecD[UInt], out offset: UInt):
        
        alias d_names = InlineArray[String, 3]("x", "y", "z")

        offset = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(coord < size, d_names[d], "=", coord, " out of range [0,", size, ")")
            offset += coord*self._strides[d]

    fn __getitem__(self, i: Self.VecD[UInt], out v: T):
        v = (self._start() + self._offset(i))[].copy()

    fn __getitem__(self, *, x: UInt, out v: T):
        expect_at_least_rank[dim, 1]()
        v = self[Self.VecD[UInt](x=x)]

    fn __getitem__(self, *, x: UInt, y: UInt, out v: T):
        expect_at_least_rank[dim, 2]()
        v = self[Self.VecD[UInt](x=x, y=y)]

    fn __getitem__(self, *, x: UInt, y: UInt, z: UInt, out v: T):
        expect_at_least_rank[dim, 3]()
        v = self[Self.VecD[UInt](x=x, y=y, z=z)]

    fn __setitem__(mut self, i: Self.VecD[UInt], v: T):
        (self._start() + self._offset(i))[] = v.copy()

    fn __setitem__(mut self, *, x: UInt, v: T):
        expect_at_least_rank[dim, 1]()
        self[Self.VecD[UInt](x=x)] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, v: T):
        expect_at_least_rank[dim, 2]()
        self[Self.VecD[UInt](x=x, y=y)] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, z: UInt, v: T):
        expect_at_least_rank[dim, 3]()
        self[Self.VecD[UInt](x=x, y=y, z=z)] = v
