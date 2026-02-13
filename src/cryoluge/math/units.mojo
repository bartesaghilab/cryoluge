
from math import sin, cos, tan, asin, acos, atan2, pi as pi_std, sqrt


@fieldwise_init
struct UnitType(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    var name: StaticString

    comptime Px = Self(1, "Px")
    comptime Ang = Self(2, "Ang")
    comptime MM = Self(3, "mm")
    comptime Rad = Self(4, "Rad")
    comptime Deg = Self(5, "Deg")
    comptime Hz = Self(6, "Hz")
    comptime KDa = Self(7, "kDa")

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)


comptime Px = Unit[UnitType.Px, _, _]
comptime Ang = Unit[UnitType.Ang, _, _]
comptime MM = Unit[UnitType.MM, _, _]
comptime Rad = Unit[UnitType.Rad, _, _]
comptime Deg = Unit[UnitType.Deg, _, _]
comptime Hz = Unit[UnitType.Hz, _, _]
comptime KDa = Unit[UnitType.KDa, _, _]

comptime pi[dtype: DType, width: Int = 1] = Rad[dtype,width](pi_std)


@register_passable("trivial")
struct Unit[
    utype: UnitType,
    dtype: DType,
    width: Int = 1
](
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Comparable,
    Writable,
    Stringable
):
    var value: Self.V

    comptime V = SIMD[dtype, width]

    @always_inline
    fn __init__(out self, value: Self.V):
        self.value = value

    @always_inline
    @implicit
    fn __init__[other_dtype: DType](out self, value: SIMD[other_dtype, width]):
        self.value = SIMD[dtype,width](value)

    @always_inline
    fn __init__[other_dtype: DType](out self, other: Unit[utype,other_dtype]):
        self.value = SIMD[dtype,width](other.value)

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.value)

    @always_inline
    fn __str__(self) -> String:
        return String.write(self)

    # comparisons

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    @always_inline
    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    @always_inline
    fn __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    @always_inline
    fn __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    @always_inline
    fn __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    # self math

    @always_inline
    fn __pos__(self) -> Self:
        return Self(+self.value)

    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.value)

    @always_inline
    fn __add__(self, other: Self) -> Self:
        return Self(self.value + other.value)

    @always_inline
    fn __iadd__(mut self, other: Self):
        self.value += other.value

    @always_inline
    fn __sub__(self, other: Self) -> Self:
        return Self(self.value - other.value)

    @always_inline
    fn __isub__(mut self, other: Self):
        self.value -= other.value

    @always_inline
    fn __mul__(self, other: Self) -> Self:
        return Self(self.value * other.value)

    @always_inline
    fn __imul__(mut self, other: Self):
        self.value *= other.value

    @always_inline
    fn __truediv__(self, other: Self) -> Self:
        return Self(self.value / other.value)

    @always_inline
    fn __itruediv__(mut self, other: Self):
        self.value /= other.value

    @always_inline
    fn __floordiv__(self, other: Self) -> Self:
        return Self(self.value // other.value)

    @always_inline
    fn __ifloordiv__(mut self, other: Self):
        self.value //= other.value
 
    @always_inline
    fn __pow__(self, other: Self) -> Self:
        return Self(self.value ** other.value)

    @always_inline
    fn __ipow__(mut self, other: Self):
        self.value = self.value ** other.value

    @always_inline
    fn clamp(self, *, min: Self) -> Self:
        return Self(math.max(self.value, min.value))

    @always_inline
    fn clamp(self, *, max: Self) -> Self:
        return Self(math.min(self.value, max.value))

    @always_inline
    fn clamp(self, *, min: Self, max: Self) -> Self:
        return self.clamp(min=min).clamp(max=max)

    # scalar math

    @always_inline
    fn __add__(self, other: Self.V) -> Self:
        return Self(self.value + other)

    @always_inline
    fn __radd__(self, other: Self.V) -> Self:
        return self + other

    @always_inline
    fn __iadd__(mut self, other: Self.V):
        self.value += other

    @always_inline
    fn __sub__(self, other: Self.V) -> Self:
        return Self(self.value - other)

    @always_inline
    fn __rsub__(self, other: Self.V) -> Self:
        return Self(other - self.value)

    @always_inline
    fn __isub__(mut self, other: Self.V):
        self.value -= other

    @always_inline
    fn __mul__(self, other: Self.V) -> Self:
        return Self(self.value * other)

    @always_inline
    fn __rmul__(self, other: Self.V) -> Self:
        return self * other

    @always_inline
    fn __imul__(mut self, other: Self.V):
        self.value *= other

    @always_inline
    fn __truediv__(self, other: Self.V) -> Self:
        return Self(self.value / other)

    @always_inline
    fn __rtruediv__(self, other: Self.V) -> Self:
        return Self(other / self.value)

    @always_inline
    fn __itruediv__(mut self, other: Self.V):
        self.value /= other

    @always_inline
    fn __floordiv__(self, other: Self.V) -> Self:
        return Self(self.value // other)

    @always_inline
    fn __rfloordiv__(self, other: Self.V) -> Self:
        return Self(other // self.value)

    @always_inline
    fn __ifloordiv__(mut self, other: Self.V):
        self.value //= other
 
    @always_inline
    fn __pow__(self, other: Self.V) -> Self:
        return Self(self.value ** other)

    @always_inline
    fn __rpow__(self, other: Self.V) -> Self:
        return Self(other ** self.value)

    @always_inline
    fn __ipow__(mut self, other: Self.V):
        self.value = self.value ** other

    @always_inline
    fn clamp(self, *, min: Self.V) -> Self:
        return Self(math.max(self.value, min))

    @always_inline
    fn clamp(self, *, max: Self.V) -> Self:
        return Self(math.min(self.value, max))

    @always_inline
    fn clamp(self, *, min: Self.V, max: Self.V) -> Self:
        return self.clamp(min=min).clamp(max=max)

    fn sqrt(self) -> Self:
        return Self(sqrt(self.value))

    fn abs(self) -> Self:
        return Self(abs(self.value))

    # TODO: dist? that works on everything but angles?
    
    # conversions
    # TODO: would these makes sense as extension functions?

    fn to_dtype[dtype_dst: DType](
        self,
        out mapped: Unit[utype,dtype_dst,width]
    ):
        mapped = Unit[utype,dtype_dst,width](SIMD[dtype_dst]( self.value ))

    fn to_ang(
        self: Px[dtype,width],
        pixel_size: SIMD[dtype,width],
        out ang: Ang[dtype,width]
    ):
        ang = Ang(self.value*pixel_size)

    fn to_ang(
        self: MM[dtype,width],
        out ang: Ang[dtype,width]
    ):
        ang = Ang(self.value*10_000_000)

    fn to_px(
        self: Ang[dtype,width],
        pixel_size: SIMD[dtype,width],
        out px: Px[dtype,width]
    ):
        px = Px(self.value/pixel_size)

    fn to_deg(
        self: Rad[dtype,width],
        out deg: Deg[dtype,width]
    ):
        deg = Deg[dtype,width](self.value*180/pi_std)
    
    fn to_rad(
        self: Deg[dtype,width],
        out rad: Rad[dtype,width]
    ):
        rad = Rad[dtype,width](self.value*pi_std/180)

    fn to_ang3(
        self: KDa[dtype,width],
        out ang: Ang[dtype,width]
    ):
        ang = Ang(self.value*1000/0.81)

    # angles

    fn _normalize[
        *,
        min: Unit[utype,dtype,1],
        min_inclusive: Bool,
        max: Unit[utype,dtype,1],
        max_inclusive: Bool,
        offset: Unit[utype,dtype,1]
    ](
        self: Unit[utype,dtype,1],
        out result: Unit[utype,dtype,1]
    ):
        result = self.copy()

        @parameter
        if min_inclusive:
            while result < min:
                result += offset
        else:
            while result <= min:
                result += offset
        
        @parameter
        if max_inclusive:
            while result > max:
                result -= offset
        else:
            while result >= max:
                result -= offset

    fn normalize_minus_pi_to_pi(
        self: Rad[dtype],
        out result: Rad[dtype]
    ):
        """Normalizes the angle to the range (-pi,pi]."""
        result = self._normalize[
            min=-pi[dtype],
            min_inclusive=False,
            max=pi[dtype],
            max_inclusive=True,
            offset=2*pi[dtype]
        ]()

    fn normalize_0_to_2pi(
        self: Rad[dtype,1],
        out result: Rad[dtype,1]
    ):
        """Normalizes the angle to the range [0,2pi)."""
        result = self._normalize[
            min=Rad[dtype](0),
            min_inclusive=True,
            max=2*pi[dtype],
            max_inclusive=False,
            offset=2*pi[dtype]
        ]()

    fn normalize(
        self: Rad[dtype,1],
        out result: Rad[dtype,1]
    ):
        """Normalizes the angle to the range (-pi,pi]."""
        result = self.normalize_minus_pi_to_pi()

    fn normalize_positive(
        self: Rad[dtype,1],
        out result: Rad[dtype,1]
    ):
        """Normalizes the angle to the range [0,2pi)."""
        result = self.normalize_0_to_2pi()

    fn normalize_minus_180_to_180(
        self: Deg[dtype],
        out result: Deg[dtype]
    ):
        """Normalizes the angle to the range (-180,180]."""
        result = self._normalize[
            min=Deg[dtype](-180),
            min_inclusive=False,
            max=Deg[dtype](180),
            max_inclusive=True,
            offset=Deg[dtype](360)
        ]()

    fn normalize_0_to_360(
        self: Deg[dtype,1],
        out result: Deg[dtype,1]
    ):
        """Normalizes the angle to the range [0,360)."""
        result = self._normalize[
            min=Deg[dtype](0),
            min_inclusive=True,
            max=Deg[dtype](360),
            max_inclusive=False,
            offset=Deg[dtype](360)
        ]()

    fn normalize(
        self: Deg[dtype,1],
        out result: Deg[dtype,1]
    ):
        """Normalizes the angle to the range (-180,180]."""
        result = self.normalize_minus_180_to_180()

    fn normalize_positive(
        self: Deg[dtype,1],
        out result: Deg[dtype,1]
    ):
        """Normalizes the angle to the range [0,360)."""
        result = self.normalize_0_to_360()

    fn dist(
        self: Rad[dtype],
        other: Rad[dtype],
        out result: Rad[dtype]
    ):
        var norm_a = self.normalize_minus_pi_to_pi()
        var norm_b = other.normalize_minus_pi_to_pi()
        var dist = abs(norm_a.value - norm_b.value)
        if dist > pi_std:
            dist = pi_std - dist
        result = Rad[dtype](dist)

    fn dist(
        self: Deg[dtype],
        other: Deg[dtype],
        out result: Deg[dtype]
    ):
        var norm_a = self.normalize_minus_180_to_180()
        var norm_b = other.normalize_minus_180_to_180()
        var dist = abs(norm_a.value - norm_b.value)
        if dist > 180:
            dist = 180 - dist
        result = Deg[dtype](dist)

    fn cos(
        self: Rad[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = cos(self.value)
    
    fn cos(
        self: Deg[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = self.to_rad().cos()

    @staticmethod
    fn acos(
        v: SIMD[dtype,width],
        out result: Unit[utype,dtype,width]
    ):
        @parameter
        if utype == UnitType.Rad:
            result = Self(acos(v))
        elif utype == UnitType.Deg:
            result = rebind[Unit[utype,dtype,width]]( Rad.acos(v).to_deg() )
        else:
            return _require_angle[Unit[utype,dtype,width]]()

    fn sin(
        self: Rad[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = sin(self.value)

    fn sin(
        self: Deg[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = self.to_rad().sin()

    @staticmethod
    fn asin(
        v: SIMD[dtype,width],
        out result: Unit[utype,dtype,width]
    ):
        @parameter
        if utype == UnitType.Rad:
            result = Self(asin(v))
        elif utype == UnitType.Deg:
            result = rebind[Unit[utype,dtype,width]]( Rad.asin(v).to_deg() )
        else:
            return _require_angle[Unit[utype,dtype,width]]()

    fn tan(
        self: Rad[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = tan(self.value)

    fn tan(
        self: Deg[dtype,width],
        out result: SIMD[dtype,width]
    ):
        result = self.to_rad().tan()

    @staticmethod
    fn atan2(
        y: SIMD[dtype,width],
        x: SIMD[dtype,width],
        out result: Unit[utype,dtype,width]
    ):
        @parameter
        if utype == UnitType.Rad:
            result = Self(atan2(y, x))
        elif utype == UnitType.Deg:
            result = rebind[Unit[utype,dtype,width]]( Rad.atan2(y, x).to_deg() )
        else:
            return _require_angle[Unit[utype,dtype,width]]()


fn _require_angle[T: AnyType]() -> T:
    constrained[False, String("Must call inverse trig functions on angle units")]()
    return abort[T]()
