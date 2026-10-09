
from std.sys.info import is_big_endian, is_little_endian


struct Endian(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable
):
    var value: Int

    comptime Big=Endian(0)
    comptime Little=Endian(1)

    @staticmethod
    def native() -> Self:
        comptime if is_big_endian():
            return Self.Big
        elif is_little_endian():
            return Self.Little
        else:
            comptime assert False, "Target architecture is neither big nor little endian"

    def __init__(out self, value: Int):
        self.value = value

    def __eq__(self, rhs: Endian) -> Bool:
        return self.value == rhs.value

    def write_to[W: Writer](self, mut writer: W):
        if self.value == Self.Big.value:
            writer.write("Big")
        elif self.value == Self.Little.value:
            writer.write("Little")
        else:
            writer.write("(unknown)")


def require_endian[endian: Endian]():
    # for, essentially, memcpy of file data into RAM to be correct,
    # the endian-ness of the file should match the endian-ness of the target architecture
    comptime assert endian == Endian.native(),
        # TODO: sometimes using template strings in `comptime assert` causes weird non-deterministic compiler bugs
        #t"Target architecture expected to be endian {endian}, but instead was {Endian.native()}"
        String("Target architecture expected to be endian ", endian, ", but instead was ", Endian.native())
