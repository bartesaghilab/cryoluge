
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

    alias Pix = Self(1, "Pix")
    alias Ang = Self(2, "Ang")
    alias MM = Self(3, "mm")
    alias Rad = Self(4, "rad")
    alias Deg = Self(5, "deg")

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)


alias Pix = Unit[UnitType.Pix, _, _]
alias PixFloat32 = Pix[DType.float32,_]

alias Ang = Unit[UnitType.Ang, _, _]
alias AngFloat32 = Ang[DType.float32,_]

alias MM = Unit[UnitType.MM, _, _]

alias Rad = Unit[UnitType.Ang, _, _]
alias Deg = Unit[UnitType.Deg, _, _]


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

    alias V = SIMD[dtype, width]

    @always_inline
    fn __init__(out self, value: Self.V):
        self.value = value

    @always_inline
    @implicit
    fn __init__[other_dtype: DType](out self, value: SIMD[other_dtype, width]):
        self.value = SIMD[dtype,width](value)

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

    # conversions
    # TODO: would these makes sense as extension functions?

    fn to_ang(
        self: Pix[dtype,width],
        pixel_size: SIMD[dtype,width],
        out ang: Ang[dtype,width]
    ):
        ang = Ang(self.value*pixel_size)

    fn to_ang(
        self: MM[dtype,width],
        out ang: Ang[dtype,width]
    ):
        ang = Ang(self.value*10_000_000)

    fn to_pix(
        self: Ang[dtype,width],
        pixel_size: SIMD[dtype,width],
        out pix: Pix[dtype,width]
    ):
        pix = Pix(self.value/pixel_size)

    fn to_deg(
        self: Rad[dtype,width],
        out deg: Deg[dtype,width]
    ):
        deg = Deg(rad_to_deg(rad=self.value))
    
    fn to_rad(
        self: Deg[dtype,width],
        out rad: Rad[dtype,width]
    ):
        rad = Rad(deg_to_rad(deg=self.value))
