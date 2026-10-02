
from std.math import sin, cos, tan, asin, acos, atan2, pi as pi_std, sqrt, floor, ceil
from std.os import abort


struct UnitType(
    TrivialRegisterPassable,
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable
):
    var value: Int
    var name: StaticString
    var _is_angle: Bool

    comptime _None = Self(0, "(none)", False)
    comptime Px = Self(1, "Px", False)
    comptime Ang = Self(2, "Ang", False)
    comptime MM = Self(3, "mm", False)
    comptime Rad = Self(4, "Rad", True)
    comptime Deg = Self(5, "Deg", True)
    comptime Hz = Self(6, "Hz", False)
    comptime KDa = Self(7, "kDa", False)

    # NOTE: `@fieldwise_init` seems to break constraint things, so use an explicit constructor
    @always_inline("builtin")
    def __init__(
        out self,
        value: Int,
        name: StaticString,
        _is_angle: Bool
    ):
        self.value = value
        self.name = name

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    # NOTE: naked field access seems break constraint things, so use an explicit accessor function
    @always_inline("builtin")
    def is_angle(self) -> Bool:
        return self._is_angle


comptime Px = Unit[UnitType.Px, _, _]
comptime Ang = Unit[UnitType.Ang, _, _]
comptime MM = Unit[UnitType.MM, _, _]
comptime Rad = Unit[UnitType.Rad, _, _]
comptime Deg = Unit[UnitType.Deg, _, _]
comptime Hz = Unit[UnitType.Hz, _, _]
comptime KDa = Unit[UnitType.KDa, _, _]

comptime pi[dtype: DType, slen: SIMDLength = 1] = Rad[dtype,slen](pi_std)


struct Unit[
    utype: UnitType,
    dtype: DType,
    slen: SIMDLength = 1
](
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Comparable,
    Writable,
    TrivialRegisterPassable
):
    var value: Self.V

    comptime V = SIMD[Self.dtype, Self.slen]
    comptime Scalar = Unit[Self.utype, Self.dtype]

    @inline(.always)
    def __init__(out self, value: Self.V):
        self.value = value

    @inline(.always)
    def __init__[other_dtype: DType](out self, value: SIMD[other_dtype, Self.slen]):
        self.value = Self.V(value)

    @inline(.always)
    @implicit
    def __init__(out self, value: IntLiteral):
        self.value = Self.V(value)

    @inline(.always)
    @implicit
    def __init__(out self, value: FloatLiteral):
        self.value = Self.V(value)

    @inline(.always)
    def __init__[other_dtype: DType](out self, other: Unit[Self.utype,other_dtype]):
        self.value = Self.V(other.value)

    @inline(.always)
    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.value)

    @inline(.always)
    def __getitem__(ref self, i: SIMDLength) -> Self.Scalar:
        return Self.Scalar(self.value[i])

    @inline(.always)
    def __getitem_param__[i: SIMDLength](self) -> Self.Scalar:
        return Self.Scalar(self.value[i])

    @inline(.always)
    def __setitem__(mut self, i: SIMDLength, v: Self.Scalar):
        self.value[i] = v.value

    @inline(.always)
    def __setitem_param__[i: SIMDLength](mut self, v: Self.Scalar):
        self.value[i] = v.value

    @inline(.always)
    def splat[l: SIMDLength](self) -> Unit[Self.utype,Self.dtype,l]
        where Self.slen == 1:

        var s = rebind[SIMD[Self.dtype,1]](self.value)
        return Unit[Self.utype,Self.dtype,l](s)

    # self comparisons

    @inline(.always)
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    @inline(.always)
    def __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    @inline(.always)
    def __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    @inline(.always)
    def __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    @inline(.always)
    def __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    # scalar comparisons

    @inline(.always)
    def __eq__(self, other: Self.V) -> Bool:
        return self.value == other

    @inline(.always)
    def __eq__(self, other: IntLiteral) -> Bool:
        return self.value == Self.V(other)

    @inline(.always)
    def __eq__(self, other: FloatLiteral) -> Bool:
        return self.value == Self.V(other)

    @inline(.always)
    def __lt__(self, other: Self.V) -> Bool:
        return self.value < other

    @inline(.always)
    def __lt__(self, other: IntLiteral) -> Bool:
        return self.value < Self.V(other)

    @inline(.always)
    def __lt__(self, other: FloatLiteral) -> Bool:
        return self.value < Self.V(other)

    @inline(.always)
    def __le__(self, other: Self.V) -> Bool:
        return self.value <= other

    @inline(.always)
    def __le__(self, other: IntLiteral) -> Bool:
        return self.value <= Self.V(other)

    @inline(.always)
    def __le__(self, other: FloatLiteral) -> Bool:
        return self.value <= Self.V(other)

    @inline(.always)
    def __gt__(self, other: Self.V) -> Bool:
        return self.value > other

    @inline(.always)
    def __gt__(self, other: IntLiteral) -> Bool:
        return self.value > Self.V(other)

    @inline(.always)
    def __gt__(self, other: FloatLiteral) -> Bool:
        return self.value > Self.V(other)

    @inline(.always)
    def __ge__(self, other: Self.V) -> Bool:
        return self.value >= other

    @inline(.always)
    def __ge__(self, other: IntLiteral) -> Bool:
        return self.value >= Self.V(other)

    @inline(.always)
    def __ge__(self, other: FloatLiteral) -> Bool:
        return self.value >= Self.V(other)

    # self math

    @inline(.always)
    def __pos__(self) -> Self:
        return Self(+self.value)

    @inline(.always)
    def __neg__(self) -> Self:
        return Self(-self.value)

    @inline(.always)
    def __add__(self, other: Self) -> Self:
        return Self(self.value + other.value)

    @inline(.always)
    def __iadd__(mut self, other: Self):
        self.value += other.value

    @inline(.always)
    def __sub__(self, other: Self) -> Self:
        return Self(self.value - other.value)

    @inline(.always)
    def __isub__(mut self, other: Self):
        self.value -= other.value

    @inline(.always)
    def __mul__(self, other: Self) -> Self:
        return Self(self.value * other.value)

    @inline(.always)
    def __imul__(mut self, other: Self):
        self.value *= other.value

    @inline(.always)
    def __truediv__(self, other: Self) -> Self:
        return Self(self.value / other.value)

    @inline(.always)
    def __itruediv__(mut self, other: Self):
        self.value /= other.value

    @inline(.always)
    def __floordiv__(self, other: Self) -> Self:
        return Self(self.value // other.value)

    @inline(.always)
    def __ifloordiv__(mut self, other: Self):
        self.value //= other.value
 
    @inline(.always)
    def __pow__(self, other: Self) -> Self:
        return Self(self.value ** other.value)

    @inline(.always)
    def __ipow__(mut self, other: Self):
        self.value = self.value ** other.value

    @inline(.always)
    def clamp(self, *, min: Self) -> Self:
        return Self(std.math.max(self.value, min.value))

    @inline(.always)
    def clamp(self, *, max: Self) -> Self:
        return Self(std.math.min(self.value, max.value))

    @inline(.always)
    def clamp(self, *, min: Self, max: Self) -> Self:
        return self.clamp(min=min).clamp(max=max)

    # scalar math

    @inline(.always)
    def __add__(self, other: Self.V) -> Self:
        return Self(self.value + other)

    @inline(.always)
    def __add__(self, other: IntLiteral) -> Self:
        return self + Self.V(other)

    @inline(.always)
    def __add__(self, other: FloatLiteral) -> Self:
        return self + Self.V(other)

    @inline(.always)
    def __radd__(self, other: Self.V) -> Self:
        return self + other

    @inline(.always)
    def __radd__(self, other: IntLiteral) -> Self:
        return self + Self.V(other)

    @inline(.always)
    def __radd__(self, other: FloatLiteral) -> Self:
        return self + Self.V(other)

    @inline(.always)
    def __iadd__(mut self, other: Self.V):
        self.value += other

    @inline(.always)
    def __iadd__(mut self, other: IntLiteral):
        self.value += Self.V(other)

    @inline(.always)
    def __iadd__(mut self, other: FloatLiteral):
        self.value += Self.V(other)

    @inline(.always)
    def __sub__(self, other: Self.V) -> Self:
        return Self(self.value - other)

    @inline(.always)
    def __sub__(self, other: IntLiteral) -> Self:
        return self - Self.V(other)

    @inline(.always)
    def __sub__(self, other: FloatLiteral) -> Self:
        return self - Self.V(other)

    @inline(.always)
    def __rsub__(self, other: Self.V) -> Self:
        return Self(other - self.value)

    @inline(.always)
    def __rsub__(self, other: IntLiteral) -> Self:
        return Self.V(other) - self

    @inline(.always)
    def __rsub__(self, other: FloatLiteral) -> Self:
        return Self.V(other) - self

    @inline(.always)
    def __isub__(mut self, other: Self.V):
        self.value -= other

    @inline(.always)
    def __isub__(mut self, other: IntLiteral):
        self.value -= Self.V(other)

    @inline(.always)
    def __isub__(mut self, other: FloatLiteral):
        self.value -= Self.V(other)

    @inline(.always)
    def __mul__(self, other: Self.V) -> Self:
        return Self(self.value * other)

    @inline(.always)
    def __mul__(self, other: IntLiteral) -> Self:
        return self * Self.V(other)

    @inline(.always)
    def __mul__(self, other: FloatLiteral) -> Self:
        return self * Self.V(other)

    @inline(.always)
    def __rmul__(self, other: Self.V) -> Self:
        return self * other

    @inline(.always)
    def __rmul__(self, other: IntLiteral) -> Self:
        return self * Self.V(other)

    @inline(.always)
    def __rmul__(self, other: FloatLiteral) -> Self:
        return self * Self.V(other)

    @inline(.always)
    def __imul__(mut self, other: Self.V):
        self.value *= other

    @inline(.always)
    def __imul__(mut self, other: IntLiteral):
        self.value *= Self.V(other)

    @inline(.always)
    def __imul__(mut self, other: FloatLiteral):
        self.value *= Self.V(other)

    @inline(.always)
    def __truediv__(self, other: Self.V) -> Self:
        return Self(self.value / other)

    @inline(.always)
    def __truediv__(self, other: IntLiteral) -> Self:
        return self / Self.V(other)

    @inline(.always)
    def __truediv__(self, other: FloatLiteral) -> Self:
        return self / Self.V(other)

    @inline(.always)
    def __rtruediv__(self, other: Self.V) -> Self:
        return Self(other / self.value)

    @inline(.always)
    def __rtruediv__(self, other: IntLiteral) -> Self:
        return Self.V(other) / self

    @inline(.always)
    def __rtruediv__(self, other: FloatLiteral) -> Self:
        return Self.V(other) / self

    @inline(.always)
    def __itruediv__(mut self, other: Self.V):
        self.value /= other

    @inline(.always)
    def __itruediv__(mut self, other: IntLiteral):
        self.value /= Self.V(other)

    @inline(.always)
    def __itruediv__(mut self, other: FloatLiteral):
        self.value /= Self.V(other)

    @inline(.always)
    def __floordiv__(self, other: Self.V) -> Self:
        return Self(self.value // other)

    @inline(.always)
    def __floordiv__(self, other: IntLiteral) -> Self:
        return self // Self.V(other)

    @inline(.always)
    def __floordiv__(self, other: FloatLiteral) -> Self:
        return self // Self.V(other)

    @inline(.always)
    def __rfloordiv__(self, other: Self.V) -> Self:
        return Self(other // self.value)

    @inline(.always)
    def __rfloordiv__(self, other: IntLiteral) -> Self:
        return Self.V(other) // self

    @inline(.always)
    def __rfloordiv__(self, other: FloatLiteral) -> Self:
        return Self.V(other) // self

    @inline(.always)
    def __ifloordiv__(mut self, other: Self.V):
        self.value //= other

    @inline(.always)
    def __ifloordiv__(mut self, other: IntLiteral):
        self.value //= Self.V(other)

    @inline(.always)
    def __ifloordiv__(mut self, other: FloatLiteral):
        self.value //= Self.V(other)

    @inline(.always)
    def __pow__(self, other: Self.V) -> Self:
        return Self(self.value ** other)

    @inline(.always)
    def __pow__(self, other: IntLiteral) -> Self:
        return self ** Self.V(other)

    @inline(.always)
    def __pow__(self, other: FloatLiteral) -> Self:
        return self ** Self.V(other)

    @inline(.always)
    def __rpow__(self, other: Self.V) -> Self:
        return Self(other ** self.value)

    @inline(.always)
    def __rpow__(self, other: IntLiteral) -> Self:
        return Self.V(other) ** self

    @inline(.always)
    def __rpow__(self, other: FloatLiteral) -> Self:
        return Self.V(other) ** self

    @inline(.always)
    def __ipow__(mut self, other: Self.V):
        self.value = self.value ** other

    @inline(.always)
    def __ipow__(mut self, other: IntLiteral):
        self.value = self.value ** Self.V(other)

    @inline(.always)
    def clamp(self, *, min: Self.V) -> Self:
        return Self(std.math.max(self.value, min))

    @inline(.always)
    def clamp(self, *, max: Self.V) -> Self:
        return Self(std.math.min(self.value, max))

    @inline(.always)
    def clamp(self, *, min: Self.V, max: Self.V) -> Self:
        return self.clamp(min=min).clamp(max=max)

    def sqrt(self) -> Self:
        return Self(sqrt(self.value))

    def abs(self) -> Self:
        return Self(abs(self.value))

    def floor(self) -> Self:
        return Self(floor(self.value))

    def ceil(self) -> Self:
        return Self(ceil(self.value))

    # TODO: dist? that works on everything but angles?
    
    # conversions

    def to_dtype[dtype_dst: DType](
        self,
        out mapped: Unit[Self.utype,dtype_dst,Self.slen]
    ):
        mapped = Unit[Self.utype,dtype_dst,Self.slen](SIMD[dtype_dst]( self.value ))

    def to_ang(
        self,
        pixel_size: Ang[Self.dtype,Self.slen],
        out ang: Ang[Self.dtype,Self.slen]
    ) where Self.utype == .Px:
        ang = Ang(self.value*pixel_size.value)

    def to_ang(
        self,
        out ang: Ang[Self.dtype,Self.slen]
    ) where Self.utype == .MM:
        ang = Ang(self.value*10_000_000)

    def to_px(
        self,
        pixel_size: Ang[Self.dtype,Self.slen],
        out px: Px[Self.dtype,Self.slen]
    ) where Self.utype == .Ang:
        px = Px(self.value/pixel_size.value)

    def to_deg(
        self,
        out deg: Deg[Self.dtype,Self.slen]
    ) where Self.utype == .Rad:
        deg = Deg[Self.dtype,Self.slen](self.value*180/pi_std)
    
    def to_rad(
        self,
        out rad: Rad[Self.dtype,Self.slen]
    ) where Self.utype == .Deg:
        rad = Rad[Self.dtype,Self.slen](self.value*pi_std/180)

    # angles

    def _normalize[
        *,
        min: Self,
        min_inclusive: Bool,
        max: Self,
        max_inclusive: Bool,
        offset: Self
    ](
        self,
        out result: Self
    ):

        result = self.copy()

        # TODO: could vectorize?
        comptime for i in range(Self.slen):

            comptime if min_inclusive:
                while result.value[i] < min.value[i]:
                    result.value[i] += offset.value[i]
            else:
                while result.value[i] <= min.value[i]:
                    result.value[i] += offset.value[i]
            
            comptime if max_inclusive:
                while result.value[i] > max.value[i]:
                    result.value[i] -= offset.value[i]
            else:
                while result.value[i] >= max.value[i]:
                    result.value[i] -= offset.value[i]

    def _normalize[
        *,
        min: Self.Scalar,
        min_inclusive: Bool,
        max: Self.Scalar,
        max_inclusive: Bool,
        offset: Self.Scalar
    ](
        self,
        out result: Self
    ):
        result = self._normalize[
            min=min.splat[Self.slen](),
            min_inclusive=min_inclusive,
            max=max.splat[Self.slen](),
            max_inclusive=max_inclusive,
            offset=offset.splat[Self.slen]()
        ]()

    def normalize_minus_pi_to_pi(
        self,
        out result: Self
    ) where Self.utype == .Rad:
        """Normalizes the angle to the range (-pi,pi]."""
        result = self._normalize[
            min=rebind[Self.Scalar]( -pi[Self.dtype] ),
            min_inclusive=False,
            max=rebind[Self.Scalar]( pi[Self.dtype] ),
            max_inclusive=True,
            offset=rebind[Self.Scalar]( 2*pi[Self.dtype] )
        ]()

    def normalize_0_to_2pi(
        self,
        out result: Self
    ) where Self.utype == .Rad:
        """Normalizes the angle to the range [0,2pi)."""
        result = self._normalize[
            min=rebind[Self.Scalar]( Rad[Self.dtype](0) ),
            min_inclusive=True,
            max=rebind[Self.Scalar]( 2*pi[Self.dtype] ),
            max_inclusive=False,
            offset=rebind[Self.Scalar]( 2*pi[Self.dtype] )
        ]()

    def normalize(
        self,
        out result: Self
    ) where Self.utype == .Rad:
        """Normalizes the angle to the range (-pi,pi]."""
        result = self.normalize_minus_pi_to_pi()

    def normalize_positive(
        self,
        out result: Self
    ) where Self.utype == .Rad:
        """Normalizes the angle to the range [0,2pi)."""
        result = self.normalize_0_to_2pi()

    def normalize_minus_180_to_180(
        self,
        out result: Self
    ) where Self.utype == .Deg:
        """Normalizes the angle to the range (-180,180]."""
        result = self._normalize[
            min=rebind[Self.Scalar]( Deg[Self.dtype](-180) ),
            min_inclusive=False,
            max=rebind[Self.Scalar]( Deg[Self.dtype](180) ),
            max_inclusive=True,
            offset=rebind[Self.Scalar]( Deg[Self.dtype](360) )
        ]()

    def normalize_0_to_360(
        self,
        out result: Self
    ) where Self.utype == .Deg:
        """Normalizes the angle to the range [0,360)."""
        result = self._normalize[
            min=rebind[Self.Scalar]( Deg[Self.dtype](0) ),
            min_inclusive=True,
            max=rebind[Self.Scalar]( Deg[Self.dtype](360) ),
            max_inclusive=False,
            offset=rebind[Self.Scalar]( Deg[Self.dtype](360) )
        ]()

    def normalize(
        self,
        out result: Self
    ) where Self.utype == .Deg:
        """Normalizes the angle to the range (-180,180]."""
        result = self.normalize_minus_180_to_180()

    def normalize_positive(
        self,
        out result: Self
    ) where Self.utype == .Deg:
        """Normalizes the angle to the range [0,360)."""
        result = self.normalize_0_to_360()

    def dist(
        self,
        other: Self,
        out result: Self
    ) where Self.utype == .Rad:
        var norm_a = self.normalize_minus_pi_to_pi()
        var norm_b = other.normalize_minus_pi_to_pi()
        result = Self(0)
        # TODO: could vectorize
        comptime for l in range(Self.slen):
            var dist = abs(norm_a.value[l] - norm_b.value[l])
            if dist > pi_std:
                dist = pi_std - dist
            result.value[l] = dist

    def dist(
        self,
        other: Self,
        out result: Self
    ) where Self.utype == .Deg:
        var norm_a = self.normalize_minus_180_to_180()
        var norm_b = other.normalize_minus_180_to_180()
        result = Self(0)
        comptime for l in range(Self.slen):
            var dist = abs(norm_a.value[l] - norm_b.value[l])
            if dist > 180:
                dist = 180 - dist
            result.value[l] = dist

    def cos(
        self,
        out result: Self.V
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = cos(self.value)
    
    def cos(
        self,
        out result: Self.V
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = cos(self.to_rad().value)

    @staticmethod
    def acos(
        v: Self.V,
        out result: Self
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = Self(acos(v))
    
    @staticmethod
    def acos(
        v: Self.V,
        out result: Self
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = rebind[Self]( Rad.acos(v).to_deg() )

    def sin(
        self,
        out result: Self.V
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = sin(self.value)

    def sin(
        self,
        out result: Self.V
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = sin(self.to_rad().value)

    @staticmethod
    def asin(
        v: Self.V,
        out result: Self
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = Self(asin(v))
    
    @staticmethod
    def asin(
        v: Self.V,
        out result: Self
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = rebind[Self]( Rad.asin(v).to_deg() )

    def tan(
        self,
        out result: Self.V
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = tan(self.value)

    def tan(
        self,
        out result: Self.V
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = self.to_rad().tan()

    @staticmethod
    def atan2(
        y: Self.V,
        x: Self.V,
        out result: Self
    ) where Self.utype == .Rad and Self.dtype.is_floating_point():
        result = Self(atan2(y, x))

    @staticmethod
    def atan2(
        y: Self.V,
        x: Self.V,
        out result: Self
    ) where Self.utype == .Deg and Self.dtype.is_floating_point():
        result = rebind[Self]( Rad.atan2(y, x).to_deg() )
