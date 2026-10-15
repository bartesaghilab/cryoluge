
from std.math import sqrt, floor, ceil
from std.utils.numerics import isnan

from cryoluge.math import round, sinc, Matrix
from cryoluge.math.dimension import unimplemented_dimension


struct Vec[
    dim: Int,
    dtype: DType,
    slen: SIMDLength = 1
](
    Copyable,
    Equatable,
    Writable
# TODO: Mojo's constaint system isn't quite fully-baked yet
#) where dim > 0:
):
    var _values: Self.Arr

    comptime Arr = Array[Self.V, Self.dim]
    comptime V = SIMD[Self.dtype, Self.slen]
    comptime S = Scalar[Self.dtype]
    comptime WithDim = Vec[_, Self.dtype, Self.slen]
    comptime WithDType = Vec[Self.dim, _, Self.slen]
    comptime WithSlen = Vec[Self.dim, Self.dtype, _]

    @inline(.always)
    def __init__(out self, *, uninitialized: Bool):
        self._values = Self.Arr(uninitialized=uninitialized)

    @inline(.always)
    def __init__(out self, var v: Self.Arr):
        self._values = v^

    @inline(.always)
    def __init__(out self, v: Self.Arr):
        self._values = v.copy()

    @implicit
    @inline(.always)
    def __init__[
        *,
        __literal_size__: Int
    ](
        out self,
        var *elems: Self.V,
        __list_literal__: NoneType = None,
    ):
        comptime assert __literal_size__ == Self.dim,
            String("Expected Array with ", Self.dim, " element(s) but got ", __literal_size__, " instead.")

        self = Self(uninitialized=True)
        comptime for i in range(__literal_size__):
            self._values[i] = elems[i]

    @staticmethod
    @inline(.always)
    def expect_num_arguments[count: Int]():
        comptime assert Self.dim == count,
            String("Expected ", Self.dim, " argument(s), but got ", count, " instead")

    @inline(.always)
    def __init__(out self, *, x: Self.V):
        Self.expect_num_arguments[1]()
        self = [x]

    @inline(.always)
    def __init__(out self, *, x: Self.V, y: Self.V):
        Self.expect_num_arguments[2]()
        self = [x, y]

    @inline(.always)
    def __init__(out self, *, x: Self.V, y: Self.V, z: Self.V):
        Self.expect_num_arguments[3]()
        self = [x, y, z]

    @inline(.always)
    def __init__(out self, *, x: Self.V, y: Self.V, z: Self.V, w: Self.V):
        Self.expect_num_arguments[4]()
        self = [x, y, z, w]

    @inline(.always)
    def __init__(out self, *, fill: Self.V):
        self._values = Self.Arr(fill=fill)

    @inline(.always)
    def __init__(out self, *, fill_simd: Self.S):
        self._values = Self.Arr(fill=Self.V(fill_simd))

    @staticmethod
    @inline(.always)
    def expect_at_least_dim[d: Int]():
        comptime assert d <= Self.dim,
            String("Expected dimension of ", d, " or more, but dimension was ", Self.dim, " instead")

    @inline(.always)
    def x(ref self) -> ref [self._values] Self.V:
        Self.expect_at_least_dim[1]()
        return self._values[0]

    @inline(.always)
    def y(ref self) -> ref [self._values] Self.V:
        Self.expect_at_least_dim[2]()
        return self._values[1]

    @inline(.always)
    def z(ref self) -> ref [self._values] Self.V:
        Self.expect_at_least_dim[3]()
        return self._values[2]

    @inline(.always)
    def w(ref self) -> ref [self._values] Self.V:
        Self.expect_at_least_dim[4]()
        return self._values[3]

    @inline(.always)
    def __getitem_param__[d: SIMDLength](ref self) -> ref [self._values] Self.V:
        return self._values[d]

    @inline(.always)
    def __getitem__(ref self, d: Int) -> ref [self._values] Self.V:
        return self._values[d]

    @inline(.always)
    def __getitem_param__[*, slice: Int](
        self,
        *,
        out v: Self.WithSlen[1]
    ) where slice >= 0 and slice < Self.dim:

        v = Self.WithSlen[1](uninitialized=True)
        comptime for d in range(Self.dim):
            v[d] = self[d][slice]

    @inline(.always)
    def __getitem__(
        self,
        *,
        slice: Int,
        out v: Self.WithSlen[1]
    ):
        v = Self.WithSlen[1](uninitialized=True)
        comptime for d in range(Self.dim):
            v[d] = self[d][slice]

    @inline(.always)
    def __setitem_param__[*, slice: Int](
        mut self,
        *,
        v: Self.WithSlen[1]
    ):
        comptime for d in range(Self.dim):
            self[d][slice] = v[d]

    @inline(.always)
    def __setitem__(
        mut self,
        *,
        slice: Int,
        v: Self.WithSlen[1]
    ):
        comptime for d in range(Self.dim):
            self[d][slice] = v[d]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(")
        comptime for d in range(Self.dim):
            comptime if d > 0:
                writer.write(", ")
            writer.write(self._values[d])
        writer.write(")")

    # TODO: shuffle?

    @inline(.always)
    def project[count: Int](self, out result: Self.WithDim[count])
        where count > 0 and count <= Self.dim:

        result = Self.WithDim[count](uninitialized=True)
        comptime for d in range(count):
            result[d] = self[d]

    @inline(.always)
    def project_d[d: Int](self, out result: Self.WithDim[1])
        where d >= 0 and d < Self.dim:

        result = Self.WithDim[1](x=self[d])

    @inline(.always)
    def lift[
        diff_dim: Int,
        //,
        higher_dim: Int
    ](
        self,
        v: Self.WithDim[diff_dim],
        out result: Self.WithDim[higher_dim]
    ) where higher_dim > 0 and diff_dim > 0:
        comptime assert higher_dim > Self.dim,
            String("Higher dimension ", higher_dim, " must be greater than Vec dimension ", Self.dim)
        comptime exp_diff_dim = higher_dim - Self.dim
        comptime assert diff_dim == exp_diff_dim,
            String("Lifting dimension ", diff_dim, " must be ", exp_diff_dim)

        result = Self.WithDim[higher_dim](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d]
        comptime for d in range(diff_dim):
            result[Self.dim + d] = v[d]

    @inline(.always)
    def lift(self, *, y: Self.V, out result: Self.WithDim[2])
        where Self.dim == 1:

        result = self.lift[2](Self.WithDim[1]([y]))

    @inline(.always)
    def lift(self, *, y: Self.V, z: Self.V, out result: Self.WithDim[3])
        where Self.dim == 1:

        result = self.lift[3](Self.WithDim[2]([y, z]))

    @inline(.always)
    def lift(self, *, y: Self.V, z: Self.V, w: Self.V, out result: Self.WithDim[4])
        where Self.dim == 1:

        result = self.lift[4](Self.WithDim[3]([y, z, w]))

    @inline(.always)
    def lift(self, *, z: Self.V, out result: Self.WithDim[3])
        where Self.dim == 2:

        result = self.lift[3](Self.WithDim[1]([z]))

    @inline(.always)
    def lift(self, *, z: Self.V, w: Self.V, out result: Self.WithDim[4])
        where Self.dim == 2:

        result = self.lift[4](Self.WithDim[2]([z, w]))

    @inline(.always)
    def lift(self, *, w: Self.V, out result: Self.WithDim[4])
        where Self.dim == 3:

        result = self.lift[4](Self.WithDim[1]([w]))

    @inline(.always)
    def has_nan(self, out result: SIMD[.bool,Self.slen])
        where Self.dtype.is_floating_point() and Self.slen == 1:
        result = False
        comptime for d in range(Self.dim):
            result |= isnan(self[d])

    @inline(.always)
    def min(self, out result: Self.V):
        result = self[0]
        comptime for d in range(1, Self.dim):
            result = min(result, self[d])

    @inline(.always)
    def min(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = min(self[d], other[d])

    @inline(.always)
    def max(self, out result: Self.V):
        result = self[0]
        comptime for d in range(1, Self.dim):
            result = max(result, self[d])

    @inline(.always)
    def max(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = max(self[d], other[d])

    @inline(.always)
    def splat[_slen: SIMDLength](self, out result: Self.WithSlen[_slen])
        where Self.slen == 1:

        result = Self.WithSlen[_slen](uninitialized=True)
        comptime for d in range(Self.dim):
            var s = rebind[Self.S]( self[d] )
            result[d] = SIMD[Self.dtype,_slen](s)

    @inline(.always)
    def select[_dtype: DType, //, w: Int](
        self,
        true_case: Self.WithDType[_dtype],
        false_case: Self.WithDType[_dtype],
        out result: Self.WithDType[_dtype]
    ) where Self.dtype == .bool:
        result = Self.WithDType[_dtype](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].select(
                true_case[d],
                false_case[d]
            )

    # math things

    @inline(.always)
    def __neg__(self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = -self[d]

    @inline(.always)
    def __add__(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d] + other[d]

    @inline(.always)
    def __add__(self, other: Self.V, out result: Self):
        result = self + Self(fill=other)

    @inline(.always)
    def __iadd__(mut self, other: Self):
        comptime for d in range(Self.dim):
            self[d] += other[d]

    @inline(.always)
    def __iadd__(mut self, other: Self.V):
        self += Self(fill=other)

    @inline(.always)
    def __sub__(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d] - other[d]

    @inline(.always)
    def __sub__(self, other: Self.V, out result: Self):
        result = self - Self(fill=other)

    @inline(.always)
    def __isub__(mut self: Self, other: Self):
        comptime for d in range(Self.dim):
            self[d] -= other[d]

    @inline(.always)
    def __isub__(mut self: Self, other: Self.V):
        self -= Self(fill=other)

    @inline(.always)
    def __mul__(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d] * other[d]

    @inline(.always)
    def __mul__(self: Self, other: Self.V, out result: Self):
        result = self * Self(fill=other)

    @inline(.always)
    def __imul__(mut self: Self, other: Self):
        comptime for d in range(Self.dim):
            self[d] *= other[d]

    @inline(.always)
    def __imul__(mut self: Self, other: Self.V):
        self *= Self(fill=other)

    @inline(.always)
    def __floordiv__(self: Self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d] // other[d]

    @inline(.always)
    def __floordiv__(self: Self, other: Self.V, out result: Self):
        result = self // Self(fill=other)

    @inline(.always)
    def __ifloordiv__(mut self: Self, other: Self):
        comptime for d in range(Self.dim):
            self[d] //= other[d]

    @inline(.always)
    def __ifloordiv__(mut self: Self, other: Self.V):
        self //= Self(fill=other)

    @inline(.always)
    def __truediv__(self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d]/other[d]

    @inline(.always)
    def __truediv__(self, other: Self.V, out result: Self):
        result = self / Self(fill=other)

    @inline(.always)
    def __itruediv__(mut self, other: Self):
        comptime for d in range(Self.dim):
            self[d] /= other[d]

    @inline(.always)
    def __itruediv__(mut self, other: Self.V):
        self /= Self(fill=other)

    @inline(.always)
    def __pow__(self: Self, other: Self.V, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d]**other

    # NOTE: __ipow__ requires integer exponents for some reason
    @inline(.always)
    def __ipow__(mut self: Self, other: Int):
        comptime for d in range(Self.dim):
            self[d] **= other

    @inline(.always)
    def __mod__(self: Self, other: Self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d] % other[d]

    @inline(.always)
    def __mod__(self: Self, other: Self.V, out result: Self):
        result = self % Self(fill=other)

    @inline(.always)
    def __imod__(mut self: Self, other: Self):
        comptime for d in range(Self.dim):
            self[d] %= other[d]

    @inline(.always)
    def __imod__(mut self: Self, other: Self.V):
        self %= Self(fill=other)

    @inline(.always)
    def __round__(self, digits: Int, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].__round__(digits)

    @inline(.always)
    def round[digits: Int](self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = round[digits](self[d])

    @inline(.always)
    def sum(self: Self, out result: Self.V):
        result = 0
        comptime for d in range(Self.dim):
            result += self[d]

    @inline(.always)
    def product(self: Self, out result: Self.V):
        result = 1
        comptime for d in range(Self.dim):
            result *= self[d]

    @inline(.always)
    def inner_product(self: Self, other: Self, out result: Self.V):
        result = 0
        comptime for d in range(Self.dim):
            result += self[d]*other[d]

    @inline(.always)
    def two_inner_products(
        self,
        v0: Self,
        v1: Self,
        out result: Self.WithDim[2]
    ):
        result = Self.WithDim[2]([
            self.inner_product(v0),
            self.inner_product(v1)
        ])

    @inline(.always)
    def three_inner_products(
        self,
        v0: Self,
        v1: Self,
        v2: Self,
        out result: Self.WithDim[3]
    ):
        result = Self.WithDim[3]([
            self.inner_product(v0),
            self.inner_product(v1),
            self.inner_product(v2)
        ])

    @inline(.always)
    def len2(self: Self, out result: Self.V):
        # NOTE: this returns a higher-precision result than the inner product, for some reason
        #       `x**2` is probably higher-precision than `x*x`
        result = (self**2).sum()

    @inline(.always)
    def len(self, out result: Self.V):
        result = sqrt(self.len2())

    @inline(.always)
    def sinc(self, out result: Self)
        where Self.dtype.is_floating_point():

        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            comptime for l in range(Self.slen):
                result[d][l] = sinc(self[d][l])

    @inline(.always)
    def abs(self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = abs(self[d])

    @inline(.always)
    def floor(self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = floor(self[d])

    @inline(.always)
    def ceil(self, out result: Self):
        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = ceil(self[d])

    # comparisons

    # number -> bool comparisons

    @inline(.always)
    def __lt__(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].lt(other[d])

    @inline(.always)
    def __le__(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].le(other[d])

    @inline(.always)
    def __gt__(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].gt(other[d])

    @inline(.always)
    def __ge__(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].ge(other[d])

    # aggregated number -> bool comparisons

    @inline(.always)
    def __eq__(self, other: Self, out result: Bool):
        result = self.eq_all(other)

    @inline(.always)
    def eq_all(self, other: Self, out result: Bool):
        result = True
        comptime for d in range(Self.dim):
            result = result and self._values[d] == other._values[d]

    @inline(.always)
    def eq_all_simd(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](fill_simd=True)
        comptime for d in range(Self.dim):
            result[d] &= self._values[d].eq(other._values[d])

    @inline(.always)
    def eq_any(self, other: Self, out result: Bool):
        result = False
        comptime for d in range(Self.dim):
            result |= self._values[d] == other._values[d]

    @inline(.always)
    def eq_any_simd(self, other: Self, out result: Self.WithDType[.bool]):
        result = Self.WithDType[.bool](fill=False)
        comptime for d in range(Self.dim):
            result[d] |= self._values[d].eq(other._values[d])

    @inline(.always)
    def ne_all(self, other: Self, out result: Bool):
        result = ~self.eq_any(other)

    @inline(.always)
    def ne_all_simd(self, other: Self, out result: Self.WithDType[.bool]):
        result = ~self.eq_any_simd(other)

    @inline(.always)
    def ne_any(self, other: Self, out result: Bool):
        result = ~self.eq_all(other)

    @inline(.always)
    def ne_any_simd(self, other: Self, out result: Self.WithDType[.bool]):
        result = ~self.eq_all_simd(other)

    @inline(.always)
    def lt_any(self: Self, other: Self, out result: Bool):
        result = False
        comptime for d in range(Self.dim):
            result |= self[d] < other[d]

    @inline(.always)
    def lt_all(self: Self, other: Self, out result: Bool):
        result = ~self.ge_any(other)

    @inline(.always)
    def le_any(self: Self, other: Self, out result: Bool):
        result = False
        comptime for d in range(Self.dim):
            result |= self[d] <= other[d]

    @inline(.always)
    def le_all(self: Self, other: Self, out result: Bool):
        result = ~self.gt_any(other)

    @inline(.always)
    def gt_any(self: Self, other: Self, out result: Bool):
        result = False
        comptime for d in range(Self.dim):
            result |= self[d] > other[d]

    @inline(.always)
    def gt_all(self: Self, other: Self, out result: Bool):
        result = ~self.le_any(other)

    @inline(.always)
    def ge_any(self: Self, other: Self, out result: Bool):
        result = False
        comptime for d in range(Self.dim):
            result |= self[d] >= other[d]

    @inline(.always)
    def ge_all(self: Self, other: Self, out result: Bool):
        result = ~self.lt_any(other)

    # boolean logic

    @inline(.always)
    def __invert__(self, out result: Self)
        where Self.dtype == .bool:

        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = ~self[d]

    @inline(.always)
    def __and__(self, other: Self, out result: Self)
        where Self.dtype == .bool:

        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].__and__(other[d])

    @inline(.always)
    def __iand__(mut self, other: Self)
        where Self.dtype == .bool:

        comptime for d in range(Self.dim):
            self[d] &= other[d]

    @inline(.always)
    def __or__(self, other: Self, out result: Self)
        where Self.dtype == .bool:

        result = Self(uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = self[d].__or__(other[d])

    @inline(.always)
    def __ior__(mut self, other: Self)
        where Self.dtype == .bool:

        comptime for d in range(Self.dim):
            self[d] |= other[d]

    # matrix interop
    
    @inline(.always)
    def column_matrix(self, out mat: Matrix[Self.dim,1,Self.dtype,Self.slen]):
        mat = Matrix[Self.dim,1,Self.dtype,Self.slen](uninitialized=True)
        comptime for d in range(Self.dim):
            mat[d,0] = self[d]

    @inline(.always)
    def row_matrix(self, out mat: Matrix[1,Self.dim,Self.dtype,Self.slen]):
        mat = Matrix[1,Self.dim,Self.dtype,Self.slen](uninitialized=True)
        comptime for d in range(Self.dim):
            mat[0,d] = self[d]

    # mappings

    @inline(.always)
    def map[
        _dtype: DType,
        _slen: SIMDLength,
        //
    ](
        self,
        f: Some[def(Self.V) -> SIMD[_dtype,_slen]],
        out result: Vec[Self.dim,_dtype,_slen]
    ):
        result = Vec[Self.dim,_dtype,_slen](uninitialized=True)
        comptime for d in range(Self.dim):
            result[d] = f(self[d])

    @inline(.always)
    def map_dtype[_dtype: DType](self, out result: Self.WithDType[_dtype]):
        result = self.map(lambda (v: Self.V) -> SIMD[_dtype,Self.slen]: SIMD[_dtype,Self.slen](v))

    def iterate_over_sizes[l: Int = 0](
        self,
        f: Some[def (i: Self)]
    ) where Self.dtype == .int:

        comptime if Self.dim == 1:
            
            for _x in range(self.x()[l]):
                var x = rebind[Self.V](_x)
                f([x])

        elif Self.dim == 2:
            
            for _y in range(self.y()[l]):
                var y = rebind[Self.V](_y)
                for _x in range(self.x()[l]):
                    var x = rebind[Self.V](_x)
                    f([x, y])

        elif Self.dim == 3:

            for _z in range(self.z()[l]):
                var z = rebind[Self.V](_z)
                for _y in range(self.y()[l]):
                    var y = rebind[Self.V](_y)
                    for _x in range(self.x()[l]):
                        var x = rebind[Self.V](_x)
                        f([x, y, z])

        else:
            unimplemented_dimension[Self.dim]()
