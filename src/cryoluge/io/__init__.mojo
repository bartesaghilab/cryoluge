
from .buffers import *
from .endian import *
from .binary_writer import *
from .binary_reader import *
from .bytes_writer import *
from .bytes_reader import *
from .file_writer import *
from .file_reader import *


fn as_byte_span[
    dtype: DType,
    origin: Origin
](ref [origin] v: Scalar[dtype]) -> Span[Byte, origin]:
    var p: UnsafePointer[Byte, origin=origin] =
        UnsafePointer(to=v).bitcast[Byte]()
    # NOTE: mojoc seems to struggle with the type inference here,
    #       so we need to explicitly write out the pointer type to help it along
    return Span(ptr=p, length=size_of[dtype]())


# Can't use stdlib's swap() in spans because of aliasing rules (right?)
# So just implement a basic swap() instead
fn swap_in_span(s: Span[mut=True, Byte], i1: Int, i2: Int):
    debug_assert(i1 < len(s))
    debug_assert(i2 < len(s))
    var swap = s[i1]
    s[i1] = s[i2]
    s[i2] = swap


fn swap_bytes_if_needed[dtype: DType, endian: Endian](mut v: Scalar[dtype]):
    from bit import byte_swap
    @parameter
    if endian != Endian.native():
        @parameter
        if dtype.is_integral():
            v = byte_swap(v)
        else:
            # no fancy intrinsic fn for non-int types, so just do it the hard way
            var s = as_byte_span(v)
            comptime size = size_of[dtype]()
            @parameter
            if size == 4:
                swap_in_span(s, 0, 3)
                swap_in_span(s, 1, 2)
            elif size == 8:
                swap_in_span(s, 0, 7)
                swap_in_span(s, 1, 6)
                swap_in_span(s, 2, 5)
                swap_in_span(s, 3, 4)
            else:
                constrained[False, String("Can't byte swap scalar type of size ", size)]()
