
from memory import UnsafePointer, memcpy, alloc


struct ByteBuffer(Copyable, Movable):
    var _p: UnsafePointer[Byte, MutOrigin.external]
    var _size: Int
    var _alignment: Int

    fn __init__(out self, size: Int, *, alignment: Optional[Int] = None):
        debug_assert(size >= 0, "Size can't be negative: ", size)
        self._size = size
        self._alignment = alignment.or_else(1)
        self._p = alloc[Byte](size, alignment=self._alignment)

    fn __del__(deinit self):
        self._p.free()

    fn __copyinit__(out self, other: Self):
        self._size = other._size
        self._alignment = other._alignment
        self._p = alloc[Byte](self._size, alignment=self._alignment)
        memcpy[Byte](src=other._p, dest=self._p, count=self._size)

    fn size(self) -> Int:
        return self._size

    fn alignment(self) -> Int:
        return self._alignment

    fn span[origin: Origin](ref [origin] self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte, origin]:

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

        return Span(
            ptr=(self._p + start)
                .unsafe_mut_cast[origin.mut]()
                .unsafe_origin_cast[origin](),
            length=length.or_else(self._size - start)
        )
