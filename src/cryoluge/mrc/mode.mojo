
@fieldwise_init
struct Mode(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: UInt32
    var name: StaticString
    var dtype: Optional[DType]

    alias Int8 = Self(0, "Int8", DType.int8)
    alias Int16 = Self(1, "Int16", DType.int16)
    alias Float32 = Self(2, "Float32", DType.float32)
    alias ComplexInt16 = Self(3, "ComplexInt16", None)
    alias ComplexFloat32 = Self(4, "ComplexFloat32", None)
    alias UInt16 = Self(6, "UInt16", DType.uint16)
    alias Float16 = Self(12, "Float16", DType.float16)
    alias Int4 = Self(101, "Int4", None)

    @staticmethod
    fn unknown(value: UInt32) -> Self:
        return Self(value, "?", None)

    @staticmethod
    fn find(value: UInt32) -> Self:
        if value == Self.Int8.value:
            return Self.Int8
        elif value == Self.Int16.value:
            return Self.Int16
        elif value == Self.Float32.value:
            return Self.Float32
        elif value == Self.ComplexInt16.value:
            return Self.ComplexInt16
        elif value == Self.ComplexFloat32.value:
            return Self.ComplexFloat32
        elif value == Self.UInt16.value:
            return Self.UInt16
        elif value == Self.Float16.value:
            return Self.Float16
        elif value == Self.Int4.value:
            return Self.Int4
        else:
            return Self.unknown(value)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    fn __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.value, ")")
        
    fn __str__(self) -> String:
        return String.write(self)
