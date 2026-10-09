
from std.sys import size_of
from std.bit import byte_swap


# TODO
from .buffers import *
from .endian import *
# from .binary_writer import *
from .binary_reader import *
# from .bytes_writer import *
from .bytes_reader import *
# from .file_writer import *
# from .file_reader import *


def as_byte_span[
    dtype: DType,
    origin: Origin
](ref [origin] v: Scalar[dtype]) -> Span[Byte,origin]:
    return Span(
        unsafe_ptr=Pointer(to=v).unsafe_bitcast[Byte](),
        length=size_of[dtype]()
    )


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
