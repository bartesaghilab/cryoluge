
from math import sqrt, floor, ceil
from utils.numerics import isnan

from cryoluge.math import unrecognized_dimension, round
from cryoluge.math.units import Unit, UnitType, Ang, Px


comptime _TBounds = Copyable & Movable & Equatable & Writable & Stringable


struct Vec[
    dim: Int,
    T: _TBounds
](
    Copyable,
    Movable,
    Equatable,
    Writable,
    Stringable
):
    var _values: InlineArray[T,dim]

    @inline(.always)
    def __init__(out self, v: InlineArray[T,dim]):
        self._values = v

    @inline(.always)
    def __init__(out self, *, x: T):
        expect_num_arguments[dim, 1]()
        self._values = InlineArray[T,dim](x.copy())

    @inline(.always)
    def __init__(out self, *, x: T, y: T):
        expect_num_arguments[dim, 2]()
        self._values = InlineArray[T,dim](x.copy(), y.copy())

    @inline(.always)
    def __init__(out self, *, x: T, y: T, z: T):
        expect_num_arguments[dim, 3]()
        self._values = InlineArray[T,dim](x.copy(), y.copy(), z.copy())

    @inline(.always)
    def __init__(out self, *, x: T, y: T, z: T, w: T):
        expect_num_arguments[dim, 4]()
        self._values = InlineArray[T,dim](x.copy(), y.copy(), z.copy(), w.copy())

    # tragically, there seems to be no way to `constrained` the length of a varargs (ie, `*v: T`),
    # so these overloads seem like the best we can do for now for higher-dimensional initializers

    @inline(.always)
    def __init__(out self, *, d1: T, d2: T, d3: T, d4: T, d5: T):
        expect_num_arguments[dim, 5]()
        self._values = InlineArray[T,dim](d1.copy(), d2.copy(), d3.copy(), d4.copy(), d5.copy())

    @inline(.always)
    def __init__(out self, *, d1: T, d2: T, d3: T, d4: T, d5: T, d6: T):
        expect_num_arguments[dim, 6]()
        self._values = InlineArray[T,dim](d1.copy(), d2.copy(), d3.copy(), d4.copy(), d5.copy(), d6.copy())

    @inline(.always)
    def __init__(out self, *, d1: T, d2: T, d3: T, d4: T, d5: T, d6: T, d7: T):
        expect_num_arguments[dim, 7]()
        self._values = InlineArray[T,dim](d1.copy(), d2.copy(), d3.copy(), d4.copy(), d5.copy(), d6.copy(), d7.copy())

    @inline(.always)
    def __init__(out self, *, d1: T, d2: T, d3: T, d4: T, d5: T, d6: T, d7: T, d8: T):
        expect_num_arguments[dim, 8]()
        self._values = InlineArray[T,dim](d1.copy(), d2.copy(), d3.copy(), d4.copy(), d5.copy(), d6.copy(), d7.copy(), d8.copy())

    @inline(.always)
    def __init__(out self, *, d1: T, d2: T, d3: T, d4: T, d5: T, d6: T, d7: T, d8: T, d9: T):
        expect_num_arguments[dim, 9]()
        self._values = InlineArray[T,dim](d1.copy(), d2.copy(), d3.copy(), d4.copy(), d5.copy(), d6.copy(), d7.copy(), d8.copy(), d9.copy())

    @inline(.always)
    def __init__(out self, *, fill: T):
        self._values = InlineArray[T,dim](fill=fill)

    @inline(.always)
    def __init__[w: Int](out self: Vec[dim,SIMD[DType.bool,w]], *, fill: Bool):
        self._values = InlineArray[SIMD[DType.bool,w],dim](fill=SIMD[DType.bool,w](fill=fill))

    @inline(.always)
    def __init__(out self, *, uninitialized: Bool):
        self._values = InlineArray[T,dim](uninitialized=uninitialized)

    @inline(.always)
    def x(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 1]()
        return self._values[0]

    @inline(.always)
    def y(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 2]()
        return self._values[1]

    @inline(.always)
    def z(ref self) -> ref [self._values] T:
        expect_at_least_rank[dim, 3]()
        return self._values[2]

    @inline(.always)
    def __getitem__(self, d: Int, out v: T):
        v = self._values[d].copy()

    @inline(.always)
    def __setitem__(mut self, d: Int, v: T):
        self._values[d] = v.copy()

    @inline(.always)
    def __getitem__[dtype: DType, simd_width: Int](
        self: Vec[dim,SIMD[dtype,simd_width]],
        *,
        slice: Int,
        out v: Vec[dim,Scalar[dtype]]
    ):
        v = Vec[dim,Scalar[dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            v[d] = self[d][slice]

    @inline(.always)
    def __setitem__[dtype: DType, simd_width: Int](
        mut self: Vec[dim,SIMD[dtype,simd_width]],
        *,
        slice: Int,
        v: Vec[dim,Scalar[dtype]]
    ):
        @parameter
        for d in range(dim):
            self[d][slice] = v[d]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(")
        @parameter
        for d in range(dim):
            @parameter
            if d > 0:
                writer.write(", ")
            writer.write(self._values[d])
        writer.write(")")

    def __str__(self) -> String:
        return String.write(self)

    @inline(.always)
    def project[pdim: Int](self, out result: Vec[pdim,T]):
        constrained[
            pdim <= dim,
            String("Projected rank ", pdim, " must be lesser or equal to vec rank ", dim)
        ]()
        result = Vec[pdim,T](uninitialized=True)
        @parameter
        for d in range(pdim):
            result[d] = self[d]

    @inline(.always)
    def project_2(self, out result: Vec[2,T]):
        return self.project[2]()

    @inline(.always)
    def project_1(self, out result: Vec[1,T]):
        return self.project[1]()

    @inline(.always)
    def select[d: Int](self, out result: Vec[1,T]):
        result = Vec[1](x=self[d])

    @inline(.always)
    def lift[
        higher_dim: Int,
        diff_dim: Int
    ](self: Vec[dim,T], v: Vec[diff_dim,T], out result: Vec[higher_dim,T]):
        constrained[
            higher_dim > dim,
            String("Lifted rank ", higher_dim, " must be higher than vec rank ", dim)
        ]()
        comptime exp_diff_dim = higher_dim - dim
        constrained[
            diff_dim == exp_diff_dim,
            String("Values dimension ", diff_dim, " must be ", exp_diff_dim)
        ]()
        result = Vec[higher_dim,T](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]
        @parameter
        for d in range(diff_dim):
            result[dim + d] = v[d]

    @inline(.always)
    def lift(self: Vec[1,T], *, y: T, out result: Vec[2,T]):
        result = self.lift[2,1](Vec[1,T](x=y))

    @inline(.always)
    def lift(self: Vec[1,T], *, y: T, z: T, out result: Vec[3,T]):
        result = self.lift[3,2](Vec[2,T](x=y, y=z))

    @inline(.always)
    def lift(self: Vec[2,T], *, z: T, out result: Vec[3,T]):
        result = self.lift[3,1](Vec[1,T](x=z))

    @inline(.always)
    def lift(self: Vec[3,T], *, w: T, out result: Vec[4,T]):
        result = self.lift[4,1](Vec[1,T](x=w))

    @inline(.always)
    def has_nan[dtype: DType](self: Vec[dim,Scalar[dtype]], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or isnan(self[d])

    @inline(.always)
    def has_nan_simd[dtype: DType, w:Int](self: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result | isnan(self[d])

    @inline(.always)
    def min(self: Vec[dim,Int], out result: Int):
        result = self[0]
        @parameter
        for d in range(1, dim):
            result = min(result, self[d])

    @inline(.always)
    def min(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = min(self[d], other[d])

    @inline(.always)
    def min[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = min(self[d], other[d])

    @inline(.always)
    def max(self: Vec[dim,Int], out result: Int):
        result = self[0]
        @parameter
        for d in range(1, dim):
            result = max(result, self[d])

    @inline(.always)
    def max(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = max(self[d], other[d])

    @inline(.always)
    def max[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = max(self[d], other[d])

    @inline(.always)
    def splat[simd_width: Int, dtype: DType](self: Vec[dim,Scalar[dtype]], out result: Vec[dim,SIMD[dtype,simd_width]]):
        result = Vec[dim,SIMD[dtype,simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = SIMD[dtype,simd_width](self[d])

    @inline(.always)
    def splat[w: Int](self: Vec[dim,Int], out result: Vec[dim,SIMD[DType.int,w]]):
        result = self.map_dint().splat[w]()

    @inline(.always)
    def select[dtype: DType, w: Int](
        self: Vec[dim,SIMD[DType.bool,w]],
        true_case: Vec[dim,SIMD[dtype,w]],
        false_case: Vec[dim,SIMD[dtype,w]],
        out result: Vec[dim,SIMD[dtype,w]]
    ):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].select(
                true_case[d],
                false_case[d]
            )

    # math things
    # NOTE: looks like we need to use conditional conformance here (eg, specialize on Int),
    #       since mojo doesn't seem to have traits for their math dunder methods =(

    # TODO: math funcs for units

    @inline(.always)
    def __neg__(self: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = -self[d]

    @inline(.always)
    def __neg__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = -self[d]

    @inline(.always)
    def __neg__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = -self[d]

    @inline(.always)
    def __add__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = self + Vec[dim,Int](fill=other)

    @inline(.always)
    def __add__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = self + Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __add__[utype: UnitType, dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self + Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __add__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self + Unit[utype,dtype](other)

    @inline(.always)
    def __iadd__(mut self: Vec[dim,Int], other: Vec[dim,Int]):
        @parameter
        for d in range(dim):
            self[d] += other[d]

    @inline(.always)
    def __iadd__(mut self: Vec[dim,Int], other: Int):
        self += Vec[dim,Int](fill=other)

    @inline(.always)
    def __iadd__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]]):
        @parameter
        for d in range(dim):
            self[d] += other[d]

    @inline(.always)
    def __iadd__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w]):
        self += Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __iadd__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]]):
        @parameter
        for d in range(dim):
            self[d] += other[d]

    @inline(.always)
    def __iadd__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]]):
        @parameter
        for d in range(dim):
            self[d] += other[d]

    @inline(.always)
    def __iadd__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype]):
        self += Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __iadd__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype]):
        self += Unit[utype,dtype](other)

    @inline(.always)
    def __sub__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] - other[d]

    @inline(.always)
    def __sub__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = self - Vec[dim,Int](fill=other)

    @inline(.always)
    def __sub__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] - other[d]

    @inline(.always)
    def __sub__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = self - Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __sub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] - other[d]

    @inline(.always)
    def __sub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] - other[d]

    @inline(.always)
    def __sub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self - Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __sub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self - Unit[utype,dtype](other)

    @inline(.always)
    def __isub__(mut self: Vec[dim,Int], other: Vec[dim,Int]):
        @parameter
        for d in range(dim):
            self[d] -= other[d]

    @inline(.always)
    def __isub__(mut self: Vec[dim,Int], other: Int):
        self -= Vec[dim,Int](fill=other)

    @inline(.always)
    def __isub__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]]):
        @parameter
        for d in range(dim):
            self[d] -= other[d]

    @inline(.always)
    def __isub__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w]):
        self -= Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __isub__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]]):
        @parameter
        for d in range(dim):
            self[d] -= other[d]

    @inline(.always)
    def __isub__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]]):
        @parameter
        for d in range(dim):
            self[d] -= other[d]

    @inline(.always)
    def __isub__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype]):
        self -= Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __isub__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype]):
        self -= Unit[utype,dtype](other)

    @inline(.always)
    def __rsub__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = other - self

    @inline(.always)
    def __rsub__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = Vec[dim,Int](fill=other) - self

    @inline(.always)
    def __rsub__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](fill=other) - self

    @inline(.always)
    def __rsub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](fill=other) - self

    @inline(.always)
    def __rsub__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = Unit[utype,dtype](other) - self

    @inline(.always)
    def __mul__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = self * Vec[dim,Int](fill=other)

    @inline(.always)
    def __mul__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,Int], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = self * Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __mul__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: IntLiteral, out result: Vec[dim,SIMD[dtype,w]]):
        result = self * SIMD[dtype,w](other)

    @inline(.always)
    def __mul__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: FloatLiteral, out result: Vec[dim,SIMD[dtype,w]]):
        result = self * SIMD[dtype,w](other)

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self * Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self * Unit[utype,dtype](other)

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: IntLiteral, out result: Vec[dim,Unit[utype,dtype]]):
        result = self * Scalar[dtype](other)

    @inline(.always)
    def __mul__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: FloatLiteral, out result: Vec[dim,Unit[utype,dtype]]):
        result = self * Scalar[dtype](other)

    @inline(.always)
    def __imul__(mut self: Vec[dim,Int], other: Vec[dim,Int]):
        @parameter
        for d in range(dim):
            self[d] *= other[d]

    @inline(.always)
    def __imul__(mut self: Vec[dim,Int], other: Int):
        self *= Vec[dim,Int](fill=other)

    @inline(.always)
    def __imul__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]]):
        @parameter
        for d in range(dim):
            self[d] *= other[d]

    @inline(.always)
    def __imul__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w]):
        self *= Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __imul__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: IntLiteral):
        self *= SIMD[dtype,w](other)

    @inline(.always)
    def __imul__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: FloatLiteral):
        self *= SIMD[dtype,w](other)

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]]):
        @parameter
        for d in range(dim):
            self[d] *= other[d]

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]]):
        @parameter
        for d in range(dim):
            self[d] *= other[d]

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype]):
        self *= Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype]):
        self *= Unit[utype,dtype](other)

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: IntLiteral):
        self *= Scalar[dtype](other)

    @inline(.always)
    def __imul__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: FloatLiteral):
        self *= Scalar[dtype](other)

    @inline(.always)
    def __floordiv__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]//other[d]

    @inline(.always)
    def __floordiv__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = self//Vec[dim,Int](fill=other)

    @inline(.always)
    def __floordiv__[w: Int](self: Vec[dim,SIMD[DType.int,w]], other: Vec[dim,SIMD[DType.int,w]], out result: Vec[dim,SIMD[DType.int,w]]):
        result = Vec[dim,SIMD[DType.int,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]//other[d]

    @inline(.always)
    def __floordiv__[w: Int](self: Vec[dim,SIMD[DType.int,w]], other: Int, out result: Vec[dim,SIMD[DType.int,w]]):
        result = self//Vec[dim,SIMD[DType.int,w]](fill=other)

    @inline(.always)
    def __ifloordiv__(mut self: Vec[dim,Int], other: Vec[dim,Int]):
        @parameter
        for d in range(dim):
            self[d] //= other[d]

    @inline(.always)
    def __ifloordiv__(mut self: Vec[dim,Int], other: Int):
        self //= Vec[dim,Int](fill=other)

    @inline(.always)
    def __truediv__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]/other[d]

    @inline(.always)
    def __truediv__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = self/Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __truediv__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: IntLiteral, out result: Vec[dim,SIMD[dtype,w]]):
        result = self/SIMD[dtype,w](other)

    @inline(.always)
    def __truediv__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: FloatLiteral, out result: Vec[dim,SIMD[dtype,w]]):
        result = self/SIMD[dtype,w](other)

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]/other[d]

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]/other[d]

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]/other[d]

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self/Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = self/Unit[utype,dtype](other)

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: IntLiteral, out result: Vec[dim,Unit[utype,dtype]]):
        result = self/Scalar[dtype](other)

    @inline(.always)
    def __truediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: FloatLiteral, out result: Vec[dim,Unit[utype,dtype]]):
        result = self/Scalar[dtype](other)

    @inline(.always)
    def __rtruediv__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](fill=other)/self

    @inline(.always)
    def __rtruediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](fill=other)/self

    @inline(.always)
    def __rtruediv__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype], out result: Vec[dim,Unit[utype,dtype]]):
        result = Unit[utype,dtype](other)/self

    @inline(.always)
    def __itruediv__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]]):
        @parameter
        for d in range(dim):
            self[d] /= other[d]

    @inline(.always)
    def __itruediv__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w]):
        self /= Vec[dim,SIMD[dtype,w]](fill=other)

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]]):
        @parameter
        for d in range(dim):
            self[d] /= other[d]

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]]):
        @parameter
        for d in range(dim):
            self[d] /= other[d]

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Unit[utype,dtype]):
        self /= Vec[dim,Unit[utype,dtype]](fill=other)

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Scalar[dtype]):
        self /= Unit[utype,dtype](other)

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: IntLiteral):
        self /= Scalar[dtype](other)

    @inline(.always)
    def __itruediv__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: FloatLiteral):
        self /= Scalar[dtype](other)

    @inline(.always)
    def __pow__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]**other

    @inline(.always)
    def __pow__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]**other

    @inline(.always)
    def __pow__[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Int, out result: Vec[dim,Unit[utype,dtype]]):
        result = Vec[dim,Unit[utype,dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d]**other

    @inline(.always)
    def __ipow__(mut self: Vec[dim,Int], other: Int):
        @parameter
        for d in range(dim):
            self[d] **= other

    @inline(.always)
    def __ipow__[dtype: DType, w: Int](mut self: Vec[dim,SIMD[dtype,w]], other: SIMD[dtype,w]):
        @parameter
        for d in range(dim):
            self[d] = self[d] ** other
            # NOTE: **= not implemented for SIMD[dtype,w] for some reason

    @inline(.always)
    def __ipow__[utype: UnitType, dtype: DType](mut self: Vec[dim,Unit[utype,dtype]], other: Int):
        @parameter
        for d in range(dim):
            self[d] **= other

    @inline(.always)
    def __mod__(self: Vec[dim,Int], other: Vec[dim,Int], out result: Vec[dim,Int]):
        result = Vec[dim,Int](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d] % other[d]

    @inline(.always)
    def __mod__(self: Vec[dim,Int], other: Int, out result: Vec[dim,Int]):
        result = self % Vec[dim,Int](fill=other)

    @inline(.always)
    def __imod__(mut self: Vec[dim,Int], other: Vec[dim,Int]):
        @parameter
        for d in range(dim):
            self[d] %= other[d]

    @inline(.always)
    def __mod__(mut self: Vec[dim,Int], other: Int):
        self %= Vec[dim,Int](fill=other)

    @inline(.always)
    def __round__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], digits: Int, out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].__round__(digits)

    @inline(.always)
    def round[digits: Int, dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        result = Vec[dim,SIMD[dtype,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = round[digits](self[d])

    @inline(.always)
    def sum(self: Vec[dim,Int], out result: Int):
        result = 0
        @parameter
        for d in range(dim):
            result += self[d]

    @inline(.always)
    def sum[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: SIMD[dtype,w]):
        result = SIMD[dtype,w](0)
        @parameter
        for d in range(dim):
            result += self[d]

    @inline(.always)
    def sum[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Unit[utype,dtype]):
        result = Unit[utype,dtype](0)
        @parameter
        for d in range(dim):
            result += self[d]

    @inline(.always)
    def product(self: Vec[dim,Int], out result: Int):
        result = 1
        @parameter
        for d in range(dim):
            result *= self[d]

    @inline(.always)
    def product[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: SIMD[dtype,w]):
        result = SIMD[dtype,w](1)
        @parameter
        for d in range(dim):
            result *= self[d]

    @inline(.always)
    def inner_product(self: Vec[dim,Int], other: Vec[dim,Int], out result: Int):
        result = 0
        @parameter
        for d in range(dim):
            result += self[d]*other[d]

    @inline(.always)
    def inner_product[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[dtype,w]):
        result = SIMD[dtype,w](0)
        @parameter
        for d in range(dim):
            result += self[d]*other[d]

    @inline(.always)
    def inner_product[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Unit[utype,dtype]], out result: Unit[utype,dtype]):
        result = Unit[utype,dtype](0)
        @parameter
        for d in range(dim):
            result += self[d]*other[d]

    @inline(.always)
    def inner_product[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], other: Vec[dim,Scalar[dtype]], out result: Unit[utype,dtype]):
        result = self.inner_product(other.map_unit[utype]())

    @inline(.always)
    def two_inner_products[dtype: DType, w: Int](
        self: Vec[dim,SIMD[dtype,w]],
        v0: Vec[dim,SIMD[dtype,w]],
        v1: Vec[dim,SIMD[dtype,w]],
        out result: Vec[2,SIMD[dtype,w]]
    ):
        result = Vec[2](
            x = self.inner_product(v0),
            y = self.inner_product(v1)
        )

    @inline(.always)
    def three_inner_products[dtype: DType, w: Int](
        self: Vec[dim,SIMD[dtype,w]],
        v0: Vec[dim,SIMD[dtype,w]],
        v1: Vec[dim,SIMD[dtype,w]],
        v2: Vec[dim,SIMD[dtype,w]],
        out result: Vec[3,SIMD[dtype,w]]
    ):
        result = Vec[3](
            x = self.inner_product(v0),
            y = self.inner_product(v1),
            z = self.inner_product(v2)
        )

    @inline(.always)
    def len2(self: Vec[dim,Int], out result: Int):
        # NOTE: this returns a higher-precision result than the inner product, for some reason
        #       `x**2` is probably higher-precision than `x*x`
        result = (self**2).sum()

    @inline(.always)
    def len2[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: SIMD[dtype,w]):
        result = (self**2).sum()

    @inline(.always)
    def len2[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Unit[utype,dtype]):
        result = (self**2).sum()
    
    @inline(.always)
    def len[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: SIMD[dtype,w]):
        result = sqrt(self.len2())

    @inline(.always)
    def len[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Unit[utype,dtype]):
        result = self.len2().sqrt()
    
    @inline(.always)
    def sinc[dtype: DType](self: Vec[dim,Scalar[dtype]], out result: Vec[dim,Scalar[dtype]]):
        result = Vec[dim,Scalar[dtype]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = sinc(self[d])

    @inline(.always)
    def abs(self: Vec[dim,Int], out result: Vec[dim,Int]):
        @parameter
        def func(i: Int) -> Int:
            return abs(i)
        result = self.map[mapper=func]()

    @inline(.always)
    def abs[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        @parameter
        def func(i: SIMD[dtype,w]) -> SIMD[dtype,w]:
            return abs(i)
        result = self.map[mapper=func]()

    @inline(.always)
    def abs[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        @parameter
        def func(i: Unit[utype,dtype]) -> Unit[utype,dtype]:
            return i.abs()
        result = self.map[mapper=func]()

    @inline(.always)
    def floor[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        @parameter
        def func(i: SIMD[dtype,w]) -> SIMD[dtype,w]:
            return floor(i)
        result = self.map[mapper=func]()

    @inline(.always)
    def floor[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        @parameter
        def func(i: Unit[utype,dtype]) -> Unit[utype,dtype]:
            return i.floor()
        result = self.map[mapper=func]()

    @inline(.always)
    def ceil[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[dtype,w]]):
        @parameter
        def func(i: SIMD[dtype,w]) -> SIMD[dtype,w]:
            return ceil(i)
        result = self.map[mapper=func]()

    @inline(.always)
    def ceil[utype: UnitType, dtype: DType](self: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        @parameter
        def func(i: Unit[utype,dtype]) -> Unit[utype,dtype]:
            return i.ceil()
        result = self.map[mapper=func]()

    # comparisons

    # number -> bool comparisons

    @inline(.always)
    def __lt__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].lt(other[d])

    @inline(.always)
    def __le__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].le(other[d])

    @inline(.always)
    def __gt__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].gt(other[d])

    @inline(.always)
    def __ge__[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].ge(other[d])

    # aggregated number -> bool comparisons

    @inline(.always)
    def __eq__(self, other: Self, out result: Bool):
        result = self.eq_all(other)

    @inline(.always)
    def eq_all(self, other: Self, out result: Bool):
        result = True
        @parameter
        for d in range(dim):
            result = result and self._values[d] == other._values[d]

    @inline(.always)
    def eq_all_simd[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=True)
        @parameter
        for d in range(dim):
            result = result & self._values[d].eq(other._values[d])

    @inline(.always)
    def eq_any(self, other: Self, out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self._values[d] == other._values[d]

    @inline(.always)
    def eq_any_simd[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result | self._values[d].eq(other._values[d])

    @inline(.always)
    def ne_all(self, other: Self, out result: Bool):
        result = not self.eq_any(other)

    @inline(.always)
    def ne_all_simd[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.eq_any_simd(other)

    @inline(.always)
    def ne_any(self, other: Self, out result: Bool):
        result = not self.eq_all(other)

    @inline(.always)
    def ne_any_simd[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.eq_all_simd(other)

    @inline(.always)
    def lt_any(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] < other[d]

    @inline(.always)
    def lt_any[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] < other[d]

    @inline(.always)
    def lt_any[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result.__or__(self[d].lt(other[d]))

    @inline(.always)
    def lt_all(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = not self.ge_any(other)

    @inline(.always)
    def lt_all[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = not self.ge_any(other)

    @inline(.always)
    def lt_all[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.ge_any(other)

    @inline(.always)
    def le_any(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] <= other[d]

    @inline(.always)
    def le_any[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] <= other[d]

    @inline(.always)
    def le_any[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result.__or__(self[d].le(other[d]))

    @inline(.always)
    def le_all(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = not self.gt_any(other)

    @inline(.always)
    def le_all[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = not self.gt_any(other)

    @inline(.always)
    def le_all[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.gt_any(other)

    @inline(.always)
    def gt_any(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] > other[d]

    @inline(.always)
    def gt_any[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] > other[d]

    @inline(.always)
    def gt_any[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result.__or__(self[d].gt(other[d]))

    @inline(.always)
    def gt_all(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = not self.le_any(other)

    @inline(.always)
    def gt_all[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = not self.le_any(other)

    @inline(.always)
    def gt_all[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.le_any(other)

    @inline(.always)
    def ge_any(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] >= other[d]

    @inline(.always)
    def ge_any[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = False
        @parameter
        for d in range(dim):
            result = result or self[d] >= other[d]

    @inline(.always)
    def ge_any[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = SIMD[DType.bool,w](fill=False)
        @parameter
        for d in range(dim):
            result = result.__or__(self[d].ge(other[d]))

    @inline(.always)
    def ge_all(self: Vec[dim,Int], other: Vec[dim,Int], out result: Bool):
        result = not self.lt_any(other)

    @inline(.always)
    def ge_all[dtype: DType](self: Vec[dim,Scalar[dtype]], other: Vec[dim,Scalar[dtype]], out result: Bool):
        result = not self.lt_any(other)

    @inline(.always)
    def ge_all[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], other: Vec[dim,SIMD[dtype,w]], out result: SIMD[DType.bool,w]):
        result = ~self.lt_any(other)

    # boolean logic

    @inline(.always)
    def __invert__[w: Int](self: Vec[dim,SIMD[DType.bool,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = ~self[d]

    @inline(.always)
    def __and__[w: Int](self: Vec[dim,SIMD[DType.bool,w]], other: Vec[dim,SIMD[DType.bool,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].__and__(other[d])

    @inline(.always)
    def __or__[w: Int](self: Vec[dim,SIMD[DType.bool,w]], other: Vec[dim,SIMD[DType.bool,w]], out result: Vec[dim,SIMD[DType.bool,w]]):
        result = Vec[dim,SIMD[DType.bool,w]](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = self[d].__or__(other[d])

    # matrix interop
    
    @inline(.always)
    def column_matrix[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out mat: Matrix[dim,1,dtype,w]):
        mat = Matrix[dim,1,dtype,w](uninitialized=True)
        @parameter
        for d in range(dim):
            mat[d,0] = self[d]

    @inline(.always)
    def row_matrix[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out mat: Matrix[1,dim,dtype,w]):
        mat = Matrix[1,dim,dtype,w](uninitialized=True)
        @parameter
        for d in range(dim):
            mat[0,d] = self[d]

    # mappings

    @inline(.always)
    def map[
        R: _TBounds, //,
        mapper: def(T) capturing -> R
    ](self, out result: Vec[dim,R]):
        result = Vec[dim,R](uninitialized=True)
        @parameter
        for d in range(dim):
            result[d] = mapper(self[d])

    @inline(.always)
    def map_int[dtype: DType](self: Vec[dim,Scalar[dtype]], out result: Vec[dim,Int]):
        @parameter
        def int(v: Scalar[dtype]) -> Int:
            return Int(v) 
        result = self.map[mapper=int]()

    @inline(.always)
    def map_scalar[dtype: DType](self: Vec[dim,Int], out result: Vec[dim,Scalar[dtype]]):
        @parameter
        def scalar(i: Int) -> Scalar[dtype]:
            return Scalar[dtype](i)
        result = self.map[mapper=scalar]()

    @inline(.always)
    def map_scalar[dtype: DType](self: Vec[dim,Bool], out result: Vec[dim,Scalar[dtype]]):
        @parameter
        def scalar(i: Bool) -> Scalar[dtype]:
            return Scalar[dtype](i)
        result = self.map[mapper=scalar]()

    @inline(.always)
    def map_scalar[dst: DType, src: DType, w: Int](self: Vec[dim,SIMD[src,w]], out result: Vec[dim,SIMD[dst,w]]):
        @parameter
        def scalar(v: SIMD[src,w]) -> SIMD[dst,w]:
            return SIMD[dst,w](v)
        result = self.map[mapper=scalar]()

    @inline(.always)
    def map_scalar[
        dtype_dst: DType,
        utype: UnitType,
        dtype_src: DType
    ](
        self: Vec[dim,Unit[utype,dtype_src]],
        out result: Vec[dim,Unit[utype,dtype_dst]]
    ):
        @parameter
        def scalar(v: Unit[utype,dtype_src]) -> Unit[utype,dtype_dst]:
            return Unit[utype,dtype_dst](v.value)
        result = self.map[mapper=scalar]()

    @inline(.always)
    def map_dint(self: Vec[dim,Int], out result: Vec[dim,Scalar[DType.int]]):
        result = self.map_scalar[DType.int]()

    @inline(.always)
    def map_dint[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.int,w]]):
        result = self.map_scalar[DType.int]()

    @inline(.always)
    def map_float32(self: Vec[dim,Int], out result: Vec[dim,Float32]):
        result = self.map_scalar[DType.float32]()

    @inline(.always)
    def map_float32[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.float32,w]]):
        result = self.map_scalar[DType.float32]()

    @inline(.always)
    def map_float64(self: Vec[dim,Int], out result: Vec[dim,Float64]):
        result = self.map_scalar[DType.float64]()

    @inline(.always)
    def map_float64[dtype: DType, w: Int](self: Vec[dim,SIMD[dtype,w]], out result: Vec[dim,SIMD[DType.float64,w]]):
        result = self.map_scalar[DType.float64]()

    @inline(.always)
    def map_unit[
        utype: UnitType,
        dtype: DType
    ](self: Vec[dim,Scalar[dtype]], out result: Vec[dim,Unit[utype,dtype]]):
        @parameter
        def m(v: Scalar[dtype]) -> Unit[utype,dtype]:
            return Unit[utype,dtype](v)
        result = self.map[mapper=m]()

    @inline(.always)
    def map_unit[
        dst_utype: UnitType,
        src_utype: UnitType,
        dtype: DType
    ](self: Vec[dim,Unit[src_utype,dtype]], out result: Vec[dim,Unit[dst_utype,dtype]]):
        @parameter
        def m(v: Unit[src_utype,dtype]) -> Unit[dst_utype,dtype]:
            return Unit[dst_utype,dtype](v.value)
        result = self.map[mapper=m]()

    @inline(.always)
    def map_value[
        dtype: DType,
        utype: UnitType
    ](self: Vec[dim,Unit[utype,dtype]], out result: Vec[dim,Scalar[dtype]]):
        @parameter
        def m(v: Unit[utype,dtype]) -> Scalar[dtype]:
            return v.value
        result = self.map[mapper=m]()

    @inline(.always)
    def map_px[dtype: DType](
        self: Vec[dim,Ang[dtype]],
        pixel_size: Ang[dtype],
        out result: Vec[dim,Px[dtype]]
    ):
        @parameter
        def m(v: Ang[dtype]) -> Px[dtype]:
            return v.to_px(pixel_size)
        result = self.map[mapper=m]()

    @inline(.always)
    def map_ang[dtype: DType](
        self: Vec[dim,Px[dtype]],
        pixel_size: Ang[dtype],
        out result: Vec[dim,Ang[dtype]]
    ):
        @parameter
        def m(v: Px[dtype]) -> Ang[dtype]:
            return v.to_ang(pixel_size)
        result = self.map[mapper=m]()

    def iterate_over_sizes[
        func: def (i: Vec[dim,Int]) capturing
    ](
        self: Vec[dim,Int]
    ):

        @parameter
        if dim == 1:
            
            for x in range(self.x()):
                func(Vec[dim,Int](x=x))

        elif dim == 2:
            
            for y in range(self.y()):
                for x in range(self.x()):
                    func(Vec[dim,Int](x=x, y=y))

        elif dim == 3:

            for z in range(self.z()):
                for y in range(self.y()):
                    for x in range(self.x()):
                        func(Vec[dim,Int](x=x, y=y, z=z))

        else:
            unrecognized_dimension[dim]()


def expect_num_arguments[dim: Int, count: Int]():
    constrained[
        dim == count,
        String(dim, " expects ", dim, " argument(s), but got ", count, " instead")
    ]()
