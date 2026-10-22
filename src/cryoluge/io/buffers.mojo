
from std.memory.alloc import unsafe_alloc


struct ByteBuffer(Movable, Copyable):
    var _p: Pointer[Byte,MutUntrackedOrigin]
    var _size: Int
    var _alignment: Optional[Int]

    def __init__(out self, size: Int, *, alignment: Optional[Int] = None):
        debug_assert(size >= 0, "Size can't be negative: ", size)
        # NOTE: the new Layout system removed support for runtime alignments =(
        #       so go back to unsafe_alloc() for now
        self._size = size
        self._alignment = alignment
        if alignment is None:
            self._p = unsafe_alloc[Byte](size)
        else:
            self._p = unsafe_alloc[Byte](size, alignment=alignment.value())
    
    def __deinit__(deinit self):
        self._p.unsafe_free()

    def __init__(out self, *, copy: Self):
        self = Self(copy._size, alignment=copy._alignment)
        self.span().copy_from(copy.span())

    @inline(.always)
    def length(self) -> Int:
        return self._size

    @inline(.always)
    def alignment(self) -> Optional[Int]:
        return self._alignment

    @inline(.always)
    def unsafe_ptr[origin: Origin](ref[origin] self) -> Pointer[Byte,origin]:
        return self._p
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()

    @inline(.always)
    def _check_index(self, i: Int):
        debug_assert(
            i >= 0 and i < self.length(),
            "Invalid index ", i, ":,  length=", self.length()
        )

    @inline(.always)
    def __getitem__(self, i: Int, out b: Byte):
        self._check_index(i)
        b = self.unsafe_ptr()[unsafe_offset=i]

    @inline(.always)
    def __setitem__(mut self, i: Int, b: Byte):
        self._check_index(i)
        self.unsafe_ptr()[unsafe_offset=i] = b

    @inline(.always)
    def span[origin: Origin](ref[origin] self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte,origin]:

        var span_len = length.or_else(self.length() - start)

        # safety first
        debug_assert(
            start >= 0 and span_len >= 0 and start + span_len <= self.length(),
            "Invalid span: start=", start, ", length=", span_len, ", buf length=", self.length()
        )

        return Span[Byte](unsafe_ptr=self.unsafe_ptr(), length=self._size)
            .unsafe_subspan(
                offset=start,
                length=span_len
            )
