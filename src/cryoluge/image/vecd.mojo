
alias _TBounds = Copyable & Movable & EqualityComparable & Writable & Stringable


struct VecD[
    T: _TBounds,
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

    fn __init__(out self, *, fill: T):
        self._values = InlineArray[T,dim.rank](fill=fill)

    fn __init__(out self, *, uninitialized: Bool):
        self._values = InlineArray[T,dim.rank](uninitialized=uninitialized)

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

    # casting methods
    # TODO: can we parameterize this more nicely somehow?

    fn cast_int(self: VecD[UInt,dim], out result: VecD[Int,dim]):
        result = VecD[Int,dim](unsafe_uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = Int(self[d])

    # math things
    # NOTE: looks like we need to use conditional conformance here (eg, specialize on Int,UInt),
    #       since mojo doesn't seem to have traits for their math dunder methods =(

    fn __floordiv__(self: VecD[UInt,dim], dividend: Int, out result: VecD[UInt,dim]):
        result = VecD[UInt,dim](unsafe_uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d]//dividend

    fn __floordiv__(self: VecD[Int,dim], dividend: Int, out result: VecD[Int,dim]):
        result = VecD[Int,dim](unsafe_uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d]//dividend

    fn __add__(self: VecD[Int,dim], other: VecD[Int,dim], out result: VecD[Int,dim]):
        result = VecD[Int,dim](unsafe_uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] + other[d]

    fn __sub__(self: VecD[Int,dim], other: VecD[Int,dim], out result: VecD[Int,dim]):
        result = VecD[Int,dim](unsafe_uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] - other[d]
