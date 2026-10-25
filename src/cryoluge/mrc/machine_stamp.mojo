
from std.sys import Endian


struct MachineStamp(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable
):
    var _b0: Byte
    var _b1: Byte

    comptime LittleEndian = Self(0x44, 0x44)
    comptime BigEndian = Self(0x11, 0x11)
    comptime NativeEndian = Self(0x44, 0x41)

    def __init__(out self, b0: Byte, b1: Byte):
        self._b0 = b0
        self._b1 = b1

    def __eq__(self, other: Self) -> Bool:
        return self._b0 == other._b0
            and self._b1 == other._b1

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        writer.write(hex(self._b0))
        writer.write(", ")
        writer.write(hex(self._b1))
        writer.write("]")

    def endian(self) raises -> Endian:
        if self == Self.LittleEndian:
            return .little
        elif self == Self.BigEndian:
            return .big
        elif self == Self.NativeEndian:
            return .native()
        else:
            raise Error(String("Can't determine endianness: machine stamp unrecognized: ", self))
