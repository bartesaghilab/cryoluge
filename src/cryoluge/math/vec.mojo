
alias _TBounds = Copyable & Movable & EqualityComparable & Writable & Stringable


struct Vec[
    T: _TBounds,
    dim: Dimension
](
    Copyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var _values: InlineArray[T,dim.rank]

    alias D1 = Vec[_,Dimension.D1]
    alias D2 = Vec[_,Dimension.D2]
    alias D3 = Vec[_,Dimension.D3]

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

    fn __getitem__(self, d: Int, out v: T):
        v = self._values[d].copy()

    fn __setitem__(mut self, d: Int, v: T):
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

    fn project[pdim: Dimension](self, out result: Vec[T,pdim]):
        constrained[
            pdim.rank <= dim.rank,
            String("Projected rank ", pdim.rank, " must be lesser or equal to vec rank ", dim.rank)
        ]()
        result = Vec[T,pdim](uninitialized=True)
        @parameter
        for d in range(pdim.rank):
            result[d] = self[d]

    fn project_2(self, out result: Self.D2[T]):
        return self.project[Dimension.D2]()

    fn project_1(self, out result: Self.D1[T]):
        return self.project[Dimension.D1]()

    fn lift[
        higher_dim: Dimension,
        diff_dim: Dimension
    ](self: Vec[T,dim], v: Vec[T,diff_dim], out result: Vec[T,higher_dim]):
        constrained[
            higher_dim.rank > dim.rank,
            String("Lifted rank ", higher_dim, " must be higher than vec rank ", dim.rank)
        ]()
        alias diff_rank = higher_dim.rank - dim.rank
        alias exp_diff_dim = Dimension(higher_dim.rank - dim.rank, "Difference")
        constrained[
            diff_dim == exp_diff_dim,
            String("Values rank ", diff_dim, " must be ", exp_diff_dim)
        ]()
        result = Vec[T,higher_dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d]
        @parameter
        for d in range(diff_dim.rank):
            result[dim.rank + d] = v[d]

    fn lift(self: Vec.D1[T], *, y: T, out result: Vec.D2[T]):
        result = self.lift[Dimension.D2,Dimension.D1](Vec.D1[T](x=y))

    fn lift(self: Vec.D1[T], *, y: T, z: T, out result: Vec.D3[T]):
        result = self.lift[Dimension.D3,Dimension.D2](Vec.D2[T](x=y, y=z))

    fn lift(self: Vec.D2[T], *, z: T, out result: Vec.D3[T]):
        result = self.lift[Dimension.D3,Dimension.D1](Vec.D1[T](x=z))

    # math things
    # NOTE: looks like we need to use conditional conformance here (eg, specialize on Int),
    #       since mojo doesn't seem to have traits for their math dunder methods =(

    fn __neg__(self: Vec[Int,dim], out result: Vec[Int,dim]):
        result = Vec[Int,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = -self[d]

    fn __neg__[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = -self[d]

    fn __add__(self: Vec[Int,dim], other: Vec[Int,dim], out result: Vec[Int,dim]):
        result = Vec[Int,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] + other[d]

    fn __add__(self: Vec[Int,dim], other: Int, out result: Vec[Int,dim]):
        result = self + Vec[Int,dim](fill=other)

    fn __add__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] + other[d]

    fn __add__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = self + Vec[Scalar[dtype],dim](fill=other)

    fn __iadd__(mut self: Vec[Int,dim], other: Vec[Int,dim]):
        @parameter
        for d in range(dim.rank):
            self[d] += other[d]

    fn __iadd__(mut self: Vec[Int,dim], other: Int):
        self += Vec[Int,dim](fill=other)

    fn __iadd__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim]):
        @parameter
        for d in range(dim.rank):
            self[d] += other[d]

    fn __iadd__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Scalar[dtype]):
        self += Vec[Scalar[dtype],dim](fill=other)

    fn __sub__(self: Vec[Int,dim], other: Vec[Int,dim], out result: Vec[Int,dim]):
        result = Vec[Int,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] - other[d]

    fn __sub__(self: Vec[Int,dim], other: Int, out result: Vec[Int,dim]):
        result = self - Vec[Int,dim](fill=other)

    fn __sub__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] - other[d]

    fn __sub__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = self - Vec[Scalar[dtype],dim](fill=other)

    fn __isub__(mut self: Vec[Int,dim], other: Vec[Int,dim]):
        @parameter
        for d in range(dim.rank):
            self[d] -= other[d]

    fn __isub__(mut self: Vec[Int,dim], other: Int):
        self -= Vec[Int,dim](fill=other)

    fn __isub__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim]):
        @parameter
        for d in range(dim.rank):
            self[d] -= other[d]

    fn __isub__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Scalar[dtype]):
        self -= Vec[Scalar[dtype],dim](fill=other)

    fn __rsub__(self: Vec[Int,dim], other: Vec[Int,dim], out result: Vec[Int,dim]):
        result = other - self

    fn __rsub__(self: Vec[Int,dim], other: Int, out result: Vec[Int,dim]):
        result = Vec[Int,dim](fill=other) - self

    fn __rsub__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = other - self

    fn __rsub__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](fill=other) - self

    fn __mul__(self: Vec[Int,dim], other: Vec[Int,dim], out result: Vec[Int,dim]):
        result = Vec[Int,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] * other[d]

    fn __mul__(self: Vec[Int,dim], other: Int, out result: Vec[Int,dim]):
        result = self * Vec[Int,dim](fill=other)

    fn __mul__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d] * other[d]

    fn __mul__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = self * Vec[Scalar[dtype],dim](fill=other)

    fn __imul__(mut self: Vec[Int,dim], other: Vec[Int,dim]):
        @parameter
        for d in range(dim.rank):
            self[d] *= other[d]

    fn __imul__(mut self: Vec[Int,dim], other: Int):
        self += Vec[Int,dim](fill=other)

    fn __imul__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim]):
        @parameter
        for d in range(dim.rank):
            self[d] *= other[d]

    fn __imul__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Scalar[dtype]):
        self += Vec[Scalar[dtype],dim](fill=other)

    fn __floordiv__(self: Vec[Int,dim], other: Vec[Int,dim], out result: Vec[Int,dim]):
        result = Vec[Int,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d]//other[d]

    fn __floordiv__(self: Vec[Int,dim], other: Int, out result: Vec[Int,dim]):
        result = self//Vec[Int,dim](fill=other)

    fn __ifloordiv__(mut self: Vec[Int,dim], other: Vec[Int,dim]):
        @parameter
        for d in range(dim.rank):
            self[d] //= other[d]

    fn __ifloordiv__(mut self: Vec[Int,dim], other: Int):
        self //= Vec[Int,dim](fill=other)

    fn __truediv__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = self[d]/other[d]

    fn __truediv__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = self/Vec[Scalar[dtype],dim](fill=other)

    fn __rtruediv__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = other[d]/self[d]

    fn __rtruediv__[dtype: DType](self: Vec[Scalar[dtype],dim], other: Scalar[dtype], out result: Vec[Scalar[dtype],dim]):
        result = Vec[Scalar[dtype],dim](fill=other)/self

    fn __itruediv__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim]):
        @parameter
        for d in range(dim.rank):
            self[d] /= other[d]

    fn __itruediv__[dtype: DType](mut self: Vec[Scalar[dtype],dim], other: Scalar[dtype]):
        self /= Vec[Scalar[dtype],dim](fill=other)

    fn sum(self: Vec[Int,dim], out result: Int):
        result = 0
        @parameter
        for d in range(dim.rank):
            result += self[d]

    fn sum[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Scalar[dtype]):
        result = Scalar[dtype](0)
        @parameter
        for d in range(dim.rank):
            result += self[d]

    fn product(self: Vec[Int,dim], out result: Int):
        result = 1
        @parameter
        for d in range(dim.rank):
            result *= self[d]

    fn product[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Scalar[dtype]):
        result = Scalar[dtype](1)
        @parameter
        for d in range(dim.rank):
            result *= self[d]

    fn inner_product(self: Vec[Int,dim], other: Vec[Int,dim], out result: Int):
        result = 0
        @parameter
        for d in range(dim.rank):
            result += self[d]*other[d]

    fn inner_product[dtype: DType](self: Vec[Scalar[dtype],dim], other: Vec[Scalar[dtype],dim], out result: Scalar[dtype]):
        result = Scalar[dtype](0)
        @parameter
        for d in range(dim.rank):
            result += self[d]*other[d]

    fn len2(self: Vec[Int,dim], out result: Int):
        result = self.inner_product(self)

    fn len2[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Scalar[dtype]):
        result = self.inner_product(self)

    fn map[
        R: _TBounds, //,
        mapper: fn(T) capturing -> R
    ](self, out result: Vec[R,dim]):
        result = Vec[R,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = mapper(self[d])

    fn map_int[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Vec[Int,dim]):
        @parameter
        fn int(v: Scalar[dtype]) -> Int:
            return Int(v) 
        result = self.map[mapper=int]()

    fn map_scalar[dtype: DType](self: Vec[Int,dim], out result: Vec[Scalar[dtype],dim]):
        @parameter
        fn scalar(i: Int) -> Scalar[dtype]:
            return Scalar[dtype](i)
        result = self.map[mapper=scalar]()

    fn map_scalar[dst: DType, src: DType](self: Vec[Scalar[src],dim], out result: Vec[Scalar[dst],dim]):
        @parameter
        fn scalar(v: Scalar[src]) -> Scalar[dst]:
            return Scalar[dst](v)
        result = self.map[mapper=scalar]()

    fn map_float32(self: Vec[Int,dim], out result: Vec[Float32,dim]):
        result = self.map_scalar[DType.float32]()

    fn map_float32[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Vec[Float32,dim]):
        result = self.map_scalar[DType.float32]()

    fn sinc(self: Vec[Float32,dim], out result: Vec[Float32,dim]):
        result = Vec[Float32,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            result[d] = sinc(self[d])

    fn abs(self: Vec[Int,dim], out result: Vec[Int,dim]):
        @parameter
        fn func(i: Int) -> Int:
            return abs(i)
        result = self.map[mapper=func]()

    fn abs[dtype: DType](self: Vec[Scalar[dtype],dim], out result: Vec[Scalar[dtype],dim]):
        @parameter
        fn func(i: Scalar[dtype]) -> Scalar[dtype]:
            return abs(i)
        result = self.map[mapper=func]()


fn expect_num_arguments[dim: Dimension, count: Int]():
    constrained[
        dim.rank == count,
        String(dim, " expects ", dim.rank, " argument(s), but got ", count, " instead")
    ]()
