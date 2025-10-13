
from memory import UnsafePointer, memcpy


struct ByteBuffer(Copyable, Movable):
    var _p: UnsafePointer[Byte]
    var _size: Int

    fn __init__(out self, size: Int):
        debug_assert(size >= 0, "Size can't be negative: ", size)
        self._p = UnsafePointer[Byte].alloc(size)
        self._size = size

    fn __del__(deinit self):
        self._p.free()

    fn __copyinit__(out self, other: Self):
        self._size = other._size
        self._p = UnsafePointer[Byte]().alloc(self._size)
        memcpy[Byte](self._p, other._p, self._size)

    fn size(self) -> Int:
        return self._size

    fn span(self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self)]]:

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
            ptr=p.origin_cast[True, MutableOrigin.cast_from[__origin_of(self)]](),
            length=length_v
        )
