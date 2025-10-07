
struct VecD[
    T: Copyable & Movable & EqualityComparable & Writable & Stringable,
    dim: ImageDimension
](
    Copyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var _values: InlineArray[T,dim.rank]

    alias D1 = VecD[_,ImageDimension.D1]
    alias D2 = VecD[_,ImageDimension.D2]
    alias D3 = VecD[_,ImageDimension.D3]

    fn __init__(out self, *, x: T):
        expect_num_arguments[dim, 1]()
        self._values = InlineArray[T,dim.rank](x.copy())

    fn __init__(out self, *, x: T, y: T):
        expect_num_arguments[dim, 2]()
        self._values = InlineArray[T,dim.rank](x.copy(), y.copy())

    fn __init__(out self, *, x: T, y: T, z: T):
        expect_num_arguments[dim, 3]()
        self._values = InlineArray[T,dim.rank](x.copy(), y.copy(), z.copy())

    fn x(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 1]()
        return self._values[0]

    fn y(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 2]()
        return self._values[1]

    fn z(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 3]()
        return self._values[2]

    fn __getitem__(self, d: UInt, out v: T):
        v = self._values[d].copy()

    fn __setitem__(mut self, d: UInt, v: T):
        self._values[d] = v.copy()

    fn __eq__(self, other: Self) -> Bool:
        @parameter
        for d in range(dim.rank):
            if self._values[d] != other._values[d]:
                return False
        return True

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("(")
        @parameter
        for d in range(dim.rank):
            @parameter
            if d > 0:
                writer.write(", ")
            writer.write(self._values[d])
        writer.write(")")

    fn __str__(self) -> String:
        return String.write(self)
