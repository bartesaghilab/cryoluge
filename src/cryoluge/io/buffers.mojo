
from memory import UnsafePointer, stack_allocation


struct ByteBuffer:
    var _p: UnsafePointer[Byte]
    var _size: UInt

    fn __init__(out self, size: UInt):
        debug_assert(size <= Int.MAX, "Size overflows Int", size)
        self._p = UnsafePointer[Byte].alloc(Int(size))
        self._size = size

    fn __del__(var self):
        self._p.free()

    fn size(self) -> UInt:
        return self._size

    fn span(self) -> Span[mut=True, Byte, MutableOrigin.cast_from[__origin_of(self)]]:
        return Span(
            ptr=self._p.origin_cast[mut=True, origin=MutableOrigin.cast_from[__origin_of(self)]](),
            length=self._size
        )
