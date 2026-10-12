
from std.sys import size_of, is_big_endian, is_little_endian
from std.bit import byte_swap

from cryoluge.io import as_byte_span


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


# Can't use stdlib's swap() in spans because of aliasing rules (right?)
# So just implement a basic swap() instead
def swap_in_span[origin: MutOrigin](s: Span[Byte,origin], i1: Int, i2: Int):
    debug_assert(i1 < len(s))
    debug_assert(i2 < len(s))
    var swap = s[i1]
    s[i1] = s[i2]
    s[i2] = swap


def swap_bytes_if_needed[dtype: DType, endian: Endian](mut v: Scalar[dtype]):

    comptime if endian != Endian.native():
        comptime if dtype.is_integral():
            v = byte_swap(v)
        else:
            # no fancy intrinsic def for non-int types, so just do it the hard way
            var s = as_byte_span(v)
            comptime size = size_of[dtype]()
            comptime if size == 4:
                swap_in_span(s, 0, 3)
                swap_in_span(s, 1, 2)
            elif size == 8:
                swap_in_span(s, 0, 7)
                swap_in_span(s, 1, 6)
                swap_in_span(s, 2, 5)
                swap_in_span(s, 3, 4)
            else:
                # TODO: sometimes using template strings in `comptime assert` causes weird non-deterministic compiler bugs
                comptime assert False, String("Can't byte swap scalar type of size: ", size)
                #comptime assert False, t"Can't byte swap scalar type of size {size}"
