
from .buffers import *
from .endian import *
from .binary_writer import *
from .binary_reader import *
from .bytes_writer import *
from .bytes_reader import *


alias ByteSpan[
    mut: Bool, //,
    origin: Origin[mut]
] = Span[
    Byte,
    origin,
    alignment=alignof[Byte]()
]
# For some reason, we need to explicitly specify Byte alignment
# even though the default alignment is supposed to be `alignof[T]()`.
# Somehow, mojoc isn't inferring it correctly

alias MutByteSpan[origin: Origin[True]] = ByteSpan[origin]


fn as_byte_span[
    mut: Bool,
    dtype: DType,
    origin: Origin[mut]
](ref [origin] v: Scalar[dtype]) -> ByteSpan[origin]:
    return Span(
        UnsafePointer(to=v)
            .bitcast[Byte]()
            .static_alignment_cast[alignof[Byte]()](),
        dtype.sizeof()
    )


# Can't use stdlib's swap() in spans because of aliasing rules (right?)
# So just implement a basic swap() instead
fn swap_in_span(s: MutByteSpan, i1: UInt, i2: UInt):
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
            @parameter
            if dtype.sizeof() == 4:
                swap_in_span(s, 0, 3)
                swap_in_span(s, 1, 2)
            elif dtype.sizeof() == 8:
                swap_in_span(s, 0, 7)
                swap_in_span(s, 1, 6)
                swap_in_span(s, 2, 5)
                swap_in_span(s, 3, 4)
            else:
                constrained[False, String("Can't byte swap scalar type of size ", dtype.sizeof())]()
