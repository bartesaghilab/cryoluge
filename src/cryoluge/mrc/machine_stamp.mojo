
struct MachineStamp(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: InlineArray[Byte, 2]

    alias LittleEndian = Self(0x44, 0x44)
    alias BigEndian = Self(0x11, 0x11)

    fn __init__(out self, a: Byte, b: Byte):
        self.value = InlineArray[Byte, 2](a, b)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.value[0] == rhs.value[0]
            and self.value[1] == rhs.value[1]

    fn __ne__(self, rhs: Self) -> Bool:
        return self.value[0] != rhs.value[0]
            or self.value[1] != rhs.value[1]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        writer.write(hex(self.value[0]))
        writer.write(", ")
        writer.write(hex(self.value[1]))
        writer.write("]")

    fn __str__(self) -> String:
        return String.write(self)

    fn endian(self) raises -> Endian:
        if self == Self.LittleEndian:
            return Endian.Little
        elif self == Self.BigEndian:
            return Endian.Big
        else:
            raise Error(String("Can't determine endianness: machine stamp unrecognized: ", self))
