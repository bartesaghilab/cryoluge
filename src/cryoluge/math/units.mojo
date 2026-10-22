
from std.math import sin, cos, tan, asin, acos, atan2, pi as pi_std, sqrt, floor, ceil
from std.os import abort

from cryoluge.math import pi, Vec


struct UnitType(
    TrivialRegisterPassable,
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable
):
    var value: Int
    var name: StaticString

    comptime px = Self(1, "Px")
    comptime ang = Self(2, "Ang")
    comptime mm = Self(3, "mm")
    comptime rad = Self(4, "Rad")
    comptime deg = Self(5, "Deg")
    comptime hz = Self(6, "Hz")
    comptime kda = Self(7, "kDa")

    # NOTE: `@fieldwise_init` seems to break constraint things, so use an explicit constructor
    @always_inline("builtin")
    def __init__(
        out self,
        value: Int,
        name: StaticString,
    ):
        self.value = value
        self.name = name


    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)


comptime Px = Unit[.px, _]
comptime Ang = Unit[.ang, _]
comptime MM = Unit[.mm, _]
comptime Rad = Unit[.rad, _]
comptime Deg = Unit[.deg, _]
comptime Hz = Unit[.hz, _]
comptime KDa = Unit[.kda, _]


struct Unit[
    utype: UnitType,
    T: Copyable & Deinitable
](
    Copyable,
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable),
    Movable,
    Deinitable,
    Writable where conforms_to(T, Writable),
    Equatable where conforms_to(T, Equatable),
    Comparable where conforms_to(T, Comparable)
):
    var value: Self.T

    comptime WithUnit = Unit[_, Self.T]
    comptime WithT = Unit[Self.utype, _]

    @implicit
    @inline(.always)
    def __init__(out self, v: Self.T):
        self.value = v.copy()

    @implicit
    @inline(.always)
    def __init__[dtype: DType, slen: SIMDLength](out self, v: IntLiteral)
        where Self.T == SIMD[dtype,slen]:

        self.value = rebind[Self.T]( SIMD[dtype,slen](v) ).copy()

    @implicit
    @inline(.always)
    def __init__[dtype: DType, slen: SIMDLength](out self, v: FloatLiteral)
        where Self.T == SIMD[dtype,slen] and dtype.is_floating_point():

        self.value = rebind[Self.T]( SIMD[dtype,slen](v) ).copy()

    @inline(.always)
    def write_to[W: Writer](self, mut writer: W)
        where conforms_to(Self.T, Writable):

        writer.write(self.value)

    # self comparisons

    @inline(.always)
    def __eq__(self, other: Self) -> Bool
        where conforms_to(Self.T, Equatable):

        return self.value == other.value

    @inline(.always)
    def __lt__(self, other: Self) -> Bool
        where conforms_to(Self.T, Comparable):

        return self.value < other.value

    @inline(.always)
    def __le__(self, other: Self) -> Bool
        where conforms_to(Self.T, Comparable):

        return self.value <= other.value

    @inline(.always)
    def __gt__(self, other: Self) -> Bool
        where conforms_to(Self.T, Comparable):

        return self.value > other.value

    @inline(.always)
    def __ge__(self, other: Self) -> Bool
        where conforms_to(Self.T, Comparable):

        return self.value >= other.value

    # specialization helpers

    @inline(.always)
    def _as_simd[dtype: DType, slen: SIMDLength](ref self) -> ref [self] Self.WithT[SIMD[dtype,slen]]
        where Self.T == SIMD[dtype,slen]:

        return rebind[Self.WithT[SIMD[dtype,slen]]]( self )

    @inline(.always)
    @staticmethod
    def _from_simd[dtype: DType, slen: SIMDLength](v: SIMD[dtype,slen]) -> Self
        where Self.T == SIMD[dtype,slen]:

        return Self(rebind[Self.T]( v ).copy())

    @inline(.always)
    def _as_vec[dim: Int, dtype: DType, slen: SIMDLength](ref self) -> ref [self] Self.WithT[Vec[dim,dtype,slen]]
        where Self.T == Vec[dim,dtype,slen]:

        return rebind[Self.WithT[Vec[dim,dtype,slen]]]( self )

    @inline(.always)
    @staticmethod
    def _from_vec[dim: Int, dtype: DType, slen: SIMDLength](v: Vec[dim,dtype,slen]) -> Self
        where Self.T == Vec[dim,dtype,slen]:

        return Self(rebind[Self.T]( v ).copy())

    # math

    @inline(.always)
    def __pos__[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](+s.value)

    @inline(.always)
    def __pos__[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](+s.value)

    @inline(.always)
    def __neg__[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](-s.value)

    @inline(.always)
    def __neg__[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](-s.value)

    @inline(.always)
    def __add__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value + o.value)

    @inline(.always)
    def __add__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value + o.value)

    @inline(.always)
    def __add__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value + other.value)

    @inline(.always)
    def __radd__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value + s.value)

    @inline(.always)
    def __radd__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](o.value + s.value)

    @inline(.always)
    def __radd__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](other.value + s.value)

    @inline(.always)
    def __iadd__[dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        s.value += o.value

    @inline(.always)
    def __iadd__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        s.value += o.value

    @inline(.always)
    def __iadd__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value += other.value

    @inline(.always)
    def __sub__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value - o.value)

    @inline(.always)
    def __sub__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value - o.value)

    @inline(.always)
    def __sub__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value - other.value)

    @inline(.always)
    def __rsub__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value - s.value)

    @inline(.always)
    def __rsub__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](o.value - s.value)

    @inline(.always)
    def __rsub__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](other.value - s.value)

    @inline(.always)
    def __isub__[dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        s.value -= o.value

    @inline(.always)
    def __isub__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        s.value -= o.value

    @inline(.always)
    def __isub__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value -= other.value

    @inline(.always)
    def __mul__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value * o.value)

    @inline(.always)
    def __mul__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value * o.value)

    @inline(.always)
    def __mul__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value * other.value)

    @inline(.always)
    def __rmul__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value * s.value)

    @inline(.always)
    def __rmul__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](o.value * s.value)

    @inline(.always)
    def __rmul__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](other.value * s.value)

    @inline(.always)
    def __imul__[dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        s.value *= o.value

    @inline(.always)
    def __imul__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        s.value *= o.value

    @inline(.always)
    def __imul__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value *= other.value

    @inline(.always)
    def __truediv__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value / o.value)

    @inline(.always)
    def __truediv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value / o.value)

    @inline(.always)
    def __truediv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value / other.value)

    @inline(.always)
    def __rtruediv__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value / s.value)

    @inline(.always)
    def __rtruediv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](o.value / s.value)

    @inline(.always)
    def __rtruediv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](other.value / s.value)

    @inline(.always)
    def __itruediv__[dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        s.value /= o.value

    @inline(.always)
    def __itruediv__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        s.value /= o.value

    @inline(.always)
    def __itruediv__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value /= other.value

    @inline(.always)
    def __floordiv__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value // o.value)

    @inline(.always)
    def __floordiv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value // o.value)

    @inline(.always)
    def __floordiv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value // other.value)

    @inline(.always)
    def __rfloordiv__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value // s.value)

    @inline(.always)
    def __rfloordiv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](o.value // s.value)

    @inline(.always)
    def __rfloordiv__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](other.value // s.value)

    @inline(.always)
    def __ifloordiv__[dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        s.value //= o.value

    @inline(.always)
    def __ifloordiv__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self)
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        ref o = other._as_vec[dim,dtype,slen]()
        s.value //= o.value

    @inline(.always)
    def __ifloordiv__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value //= other.value

    @inline(.always)
    def __pow__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](s.value ** o.value)

    @inline(.always)
    def __pow__[dim: Int, dtype: DType, slen: SIMDLength](self, other: Self.WithT[SIMD[dtype,slen]]) -> Self
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        return Self._from_vec[dim,dtype,slen](s.value ** other.value)

    @inline(.always)
    def __rpow__[dtype: DType, slen: SIMDLength](self, other: Self) -> Self
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        ref o = other._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](o.value ** s.value)

    # NOTE: __rpow__ doesn't make sense for Vec

    @inline(.always)
    def __ipow__[dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[Int])
        where Self.T == SIMD[dtype,slen]:
        
        ref s = self._as_simd[dtype,slen]()
        s.value **= other.value

    @inline(.always)
    def __ipow__[dim: Int, dtype: DType, slen: SIMDLength](mut self, other: Self.WithT[Int])
        where Self.T == Vec[dim,dtype,slen]:
        
        ref s = self._as_vec[dim,dtype,slen]()
        s.value **= other.value

    @inline(.always)
    def clamp[dtype: DType, slen: SIMDLength](self, *, min: Self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        ref _min = min._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](std.math.max(s.value, _min.value))

    @inline(.always)
    def clamp[dtype: DType, slen: SIMDLength](self, *, max: Self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        ref _max = max._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](std.math.min(s.value, _max.value))

    @inline(.always)
    def clamp[dtype: DType, slen: SIMDLength](self, *, min: Self, max: Self) -> Self
        where Self.T == SIMD[dtype,slen]:

        return self.clamp[dtype,slen](min=min).clamp[dtype,slen](max=max)

    @inline(.always)
    def sqrt[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](sqrt(s.value))

    @inline(.always)
    def abs[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](abs(s.value))

    @inline(.always)
    def floor[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](floor(s.value))

    @inline(.always)
    def ceil[dtype: DType, slen: SIMDLength](self) -> Self
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        return Self._from_simd[dtype,slen](ceil(s.value))

    # Vec accessors

    @inline(.always)
    def x[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithT[Vec[dim,dtype,slen].V]
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return s.value.x()

    @inline(.always)
    def y[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithT[Vec[dim,dtype,slen].V]
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return s.value.y()

    @inline(.always)
    def z[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithT[Vec[dim,dtype,slen].V]
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return s.value.z()

    @inline(.always)
    def w[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithT[Vec[dim,dtype,slen].V]
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        return s.value.w()

    # TODO: this breaks __setitem_param__() for some reason? compiler bug maybe?
    # @inline(.always)
    # def __getitem_param__[dim: Int, dtype: DType, slen: SIMDLength, //, d: SIMDLength](self, out v: Self.WithT[Vec[dim,dtype,slen].V])
    #     where Self.T == Vec[dim,dtype,slen]:

    #     ref s = self._as_vec[dim,dtype,slen]()
    #     v = s.value[d]

    @inline(.always)
    def __getitem__[dim: Int, dtype: DType, slen: SIMDLength](self, d: Int, out v: Self.WithT[Vec[dim,dtype,slen].V])
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        v = s.value[d]

    @inline(.always)
    def __setitem_param__[dim: Int, dtype: DType, slen: SIMDLength, //, d: SIMDLength](mut self, v: Self.WithT[Vec[dim,dtype,slen].V])
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        s.value[d] = v.value

    @inline(.always)
    def __setitem__[dim: Int, dtype: DType, slen: SIMDLength](mut self, d: Int, v: Self.WithT[Vec[dim,dtype,slen].V])
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        s.value[d] = v.value

    # conversions

    @inline(.always)
    def to_unit[utype_dst: UnitType](self) -> Self.WithUnit[utype_dst]:
        return Self.WithUnit[utype_dst]( self.value )

    @inline(.always)
    def to_dtype[
        dtype: DType,
        slen: SIMDLength,
        //,
        dtype_dst: DType,
    ](self) -> Self.WithT[SIMD[dtype_dst,slen]]
        where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        var v = SIMD[dtype_dst,slen]( s.value )
        return Self.WithT[SIMD[dtype_dst,slen]]( v )

    @inline(.always)
    def to_dtype[
        dim: Int,
        dtype: DType,
        slen: SIMDLength,
        //,
        dtype_dst: DType,
    ](self) -> Self.WithT[Vec[dim,dtype_dst,slen]]
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        var v = s.value.map_dtype[dtype_dst]()
        return Self.WithT[Vec[dim,dtype_dst,slen]]( v )

    # TODO: more Vec versions

    @inline(.always)
    def to_ang[dtype: DType, slen: SIMDLength](self, pixel_size: Unit[.ang,Scalar[dtype]]) -> Self.WithUnit[.ang]
        where Self.T == SIMD[dtype,slen] and Self.utype == .px:

        ref s = self._as_simd[dtype,slen]()
        return Self.WithUnit[.ang]._from_simd(s.value*pixel_size.value)

    @inline(.always)
    def to_ang[dim: Int, dtype: DType, slen: SIMDLength](self, pixel_size: Unit[.ang,Scalar[dtype]]) -> Self.WithUnit[.ang]
        where Self.T == Vec[dim,dtype,slen] and Self.utype == .px:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self.WithUnit[.ang]._from_vec(s.value*pixel_size.value)

    @inline(.always)
    def to_ang[dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.ang]
        where Self.T == SIMD[dtype,slen] and Self.utype == .mm:

        ref s = self._as_simd[dtype,slen]()
        return Self.WithUnit[.ang]._from_simd(s.value*10_000_000)

    @inline(.always)
    def to_ang[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.ang]
        where Self.T == Vec[dim,dtype,slen] and Self.utype == .mm:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self.WithUnit[.ang]._from_vec(s.value*10_000_000)

    @inline(.always)
    def to_px[dtype: DType, slen: SIMDLength](self, pixel_size: Self.WithUnit[.ang]) -> Self.WithUnit[.px]
        where Self.T == SIMD[dtype,slen] and Self.utype == .ang:

        ref s = self._as_simd[dtype,slen]()
        ref _pixel_size = pixel_size._as_simd[dtype,slen]()
        return Self.WithUnit[.px]._from_simd(s.value/_pixel_size.value)

    @inline(.always)
    def to_px[dim: Int, dtype: DType, slen: SIMDLength](self, pixel_size: Unit[.ang,Scalar[dtype]]) -> Self.WithUnit[.px]
        where Self.T == Vec[dim,dtype,slen] and Self.utype == .ang:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self.WithUnit[.px]._from_vec(s.value/pixel_size.value)

    @inline(.always)
    def to_deg[dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.deg]
        where Self.T == SIMD[dtype,slen] and Self.utype == .rad:

        ref s = self._as_simd[dtype,slen]()
        return Self.WithUnit[.deg]._from_simd[dtype,slen](s.value*180/pi_std)

    @inline(.always)
    def to_deg[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.deg]
        where Self.T == Vec[dim,dtype,slen] and Self.utype == .rad:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self.WithUnit[.deg]._from_vec[dim,dtype,slen](s.value*180/pi_std)

    @inline(.always)
    def to_rad[dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.rad]
        where Self.T == SIMD[dtype,slen] and Self.utype == .deg:

        ref s = self._as_simd[dtype,slen]()
        return Self.WithUnit[.rad]._from_simd[dtype,slen](s.value*pi_std/180)

    @inline(.always)
    def to_rad[dim: Int, dtype: DType, slen: SIMDLength](self) -> Self.WithUnit[.rad]
        where Self.T == Vec[dim,dtype,slen] and Self.utype == .deg:

        ref s = self._as_vec[dim,dtype,slen]()
        return Self.WithUnit[.rad]._from_vec[dim,dtype,slen](s.value*pi_std/180)

    @inline(.always)
    def len2[dim: Int, dtype: DType, slen: SIMDLength](self, out result: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        result = s.value.len2()

    @inline(.always)
    def len[dim: Int, dtype: DType, slen: SIMDLength](self, out result: Self.WithT[SIMD[dtype,slen]])
        where Self.T == Vec[dim,dtype,slen]:

        ref s = self._as_vec[dim,dtype,slen]()
        result = s.value.len()

    # angles

    def _normalize[
        dtype: DType,
        slen: SIMDLength,
        *,
        min: SIMD[dtype,slen],
        min_inclusive: Bool,
        max: SIMD[dtype,slen],
        max_inclusive: Bool,
        offset: SIMD[dtype,slen]
    ](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen]:

        ref s = self._as_simd[dtype,slen]()
        var v = s.value.copy()

        # TODO: could vectorize?
        comptime for i in range(slen):

            comptime if min_inclusive:
                while v[i] < min[i]:
                    v[i] += offset[i]
            else:
                while v[i] <= min[i]:
                    v[i] += offset[i]
            
            comptime if max_inclusive:
                while v[i] > max[i]:
                    v[i] -= offset[i]
            else:
                while v[i] >= max[i]:
                    v[i] -= offset[i]

        result = Self._from_simd[dtype,slen](v)

    def normalize_minus_pi_to_pi[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():
        """Normalizes the angle to the range (-pi,pi]."""

        result = self._normalize[
            dtype, slen,
            min=-pi_std,
            min_inclusive=False,
            max=pi_std,
            max_inclusive=True,
            offset=2*pi_std
        ]()

    def normalize_0_to_2pi[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():
        """Normalizes the angle to the range [0,2pi)."""

        result = self._normalize[
            dtype, slen,
            min=0,
            min_inclusive=True,
            max=2*pi_std,
            max_inclusive=False,
            offset=2*pi_std
        ]()

    def normalize[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():
        """Normalizes the angle to the range (-pi,pi]."""

        result = self.normalize_minus_pi_to_pi[dtype,slen]()

    def normalize_positive[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():
        """Normalizes the angle to the range [0,2pi)."""

        result = self.normalize_0_to_2pi[dtype,slen]()

    def normalize_minus_180_to_180[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg:
        """Normalizes the angle to the range (-180,180]."""

        result = self._normalize[
            dtype, slen,
            min=-180,
            min_inclusive=False,
            max=180,
            max_inclusive=True,
            offset=360
        ]()

    def normalize_0_to_360[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg:
        """Normalizes the angle to the range [0,360)."""

        result = self._normalize[
            dtype, slen,
            min=0,
            min_inclusive=True,
            max=360,
            max_inclusive=False,
            offset=360
        ]()

    def normalize[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg:
        """Normalizes the angle to the range (-180,180]."""

        result = self.normalize_minus_180_to_180[dtype,slen]()

    def normalize_positive[dtype: DType, slen: SIMDLength](
        self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg:
        """Normalizes the angle to the range [0,360)."""

        result = self.normalize_0_to_360[dtype,slen]()

    def dist[dtype: DType, slen: SIMDLength](
        self,
        other: Self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        var s = self._as_simd[dtype,slen]().normalize_minus_pi_to_pi[dtype,slen]()
        var o = other._as_simd[dtype,slen]().normalize_minus_pi_to_pi[dtype,slen]()

        var v = Scalar[dtype](0)
        # TODO: could vectorize
        comptime for l in range(slen):
            var dist = abs(s.value[l] - o.value[l])
            if dist > pi_std:
                dist = pi_std - dist
            v[l] = dist

        result = Self._from_simd[dtype,slen](v)

    def dist[dtype: DType, slen: SIMDLength](
        self,
        other: Self,
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg:

        var s = self._as_simd[dtype,slen]().normalize_minus_180_to_180[dtype,slen]()
        var o = other._as_simd[dtype,slen]().normalize_minus_180_to_180[dtype,slen]()

        var v = Scalar[dtype](0)
        # TODO: could vectorize
        comptime for l in range(slen):
            var dist = abs(s.value[l] - o.value[l])
            if dist > 180:
                dist = 180 - dist
            v[l] = dist

        result = Self._from_simd[dtype,slen](v)

    def cos[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        ref s = self._as_simd[dtype,slen]()
        result = cos(s.value)
    
    def cos[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        result = self.to_rad[dtype,slen]().cos[dtype,slen]()

    @staticmethod
    def acos[dtype: DType, slen: SIMDLength](
        v: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        result = Self._from_simd[dtype,slen](acos(v))
    
    @staticmethod
    def acos[dtype: DType, slen: SIMDLength](
        v: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        var rad = Self.WithUnit[.rad].acos(v)
        result = rebind[Self]( rad.to_deg[dtype,slen]() ).copy()
    
    def sin[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        ref s = self._as_simd[dtype,slen]()
        result = sin(s.value)
    
    def sin[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        result = self.to_rad[dtype,slen]().sin[dtype,slen]()

    @staticmethod
    def asin[dtype: DType, slen: SIMDLength](
        v: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        result = Self._from_simd[dtype,slen](asin(v))
    
    @staticmethod
    def asin[dtype: DType, slen: SIMDLength](
        v: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        var rad = Self.WithUnit[.rad].asin(v)
        result = rebind[Self]( rad.to_deg[dtype,slen]() ).copy()

    def tan[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        ref s = self._as_simd[dtype,slen]()
        result = tan(s.value)
    
    def tan[dtype: DType, slen: SIMDLength](
        self,
        out result: SIMD[dtype,slen]
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        result = self.to_rad[dtype,slen]().tan[dtype,slen]()

    @staticmethod
    def atan2[dtype: DType, slen: SIMDLength](
        y: SIMD[dtype,slen],
        x: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .rad and dtype.is_floating_point():

        result = Self._from_simd[dtype,slen](atan2(y, x))
    
    @staticmethod
    def atan2[dtype: DType, slen: SIMDLength](
        y: SIMD[dtype,slen],
        x: SIMD[dtype,slen],
        out result: Self
    ) where Self.T == SIMD[dtype,slen] and Self.utype == .deg and dtype.is_floating_point():

        var rad = Self.WithUnit[.rad].atan2(y, x)
        result = rebind[Self]( rad.to_deg[dtype,slen]() ).copy()
