
from .buffers import *
from .endian import *
from .binary_writer import *
from .binary_reader import *
from .bytes_writer import *
from .bytes_reader import *
from .file_writer import *
from .file_reader import *


def as_byte_span[
    dtype: DType,
    origin: Origin
](ref [origin] v: Scalar[dtype]) -> Span[Byte,origin]:
    return Span(
        unsafe_ptr=Pointer(to=v).unsafe_bitcast[Byte](),
        length=size_of[dtype]()
    )
