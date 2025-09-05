
from memory import UnsafePointer


struct ByteBuffer:
    var _p: UnsafePointer[Byte]
    var _size: UInt

    fn __init__(out self, size: UInt):
        debug_assert(size <= Int.MAX, "Size overflows Int", size)
        self._p = UnsafePointer[Byte].alloc(Int(size))
        self._size = size

    fn __del__(deinit self):
        self._p.free()

    fn size(self) -> UInt:
        return self._size

    fn span(self,
        *,
        start: UInt = 0,
        length: Optional[Int] = None
    ) -> Span[mut=True, Byte, MutableOrigin.cast_from[__origin_of(self)]]:

        # safety first
        debug_assert[assert_mode="safe"](
            start <= self._size,
            "Invalid span start: start=", start, ", size=", self._size
        )
        if length:
            debug_assert[assert_mode="safe"](
                start + length.value() <= self._size,
                "Invalid span length: start=", start, ", length=", length.value(), ", size=", self._size
            )

        var length_v = length.or_else(self._size - start)
        var p: UnsafePointer[Byte] = self._p + start
        return Span(
            ptr=p.origin_cast[mut=True, origin=MutableOrigin.cast_from[__origin_of(self)]](),
            length=length_v
        )
