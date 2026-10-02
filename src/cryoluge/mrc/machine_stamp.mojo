
struct MachineStamp(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable,
    Stringable
):
    var value: InlineArray[Byte, 2]

    comptime LittleEndian = Self(0x44, 0x44)
    comptime BigEndian = Self(0x11, 0x11)
    comptime NativeEndian = Self(0x44, 0x41)

    def __init__(out self, a: Byte, b: Byte):
        self.value = InlineArray[Byte, 2](a, b)

    def __eq__(self, rhs: Self) -> Bool:
        return self.value[0] == rhs.value[0]
            and self.value[1] == rhs.value[1]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        writer.write(hex(self.value[0]))
        writer.write(", ")
        writer.write(hex(self.value[1]))
        writer.write("]")

    def __str__(self) -> String:
        return String.write(self)

    def endian(self) raises -> Endian:
        if self == Self.LittleEndian:
            return Endian.Little
        elif self == Self.BigEndian:
            return Endian.Big
        elif self == Self.NativeEndian:
            return Endian.native()
        else:
            raise Error(String("Can't determine endianness: machine stamp unrecognized: ", self))
