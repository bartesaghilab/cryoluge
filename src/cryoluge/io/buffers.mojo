
from std.memory import alloc, dealloc, Layout, Allocation


struct ByteBuffer(Movable, Copyable):
    var _alloc: Allocation[Byte]

    def __init__(out self, size: Int, *, alignment: Optional[Int] = None):
        debug_assert(size >= 0, "Size can't be negative: ", size)

        var layout: Layout[Byte]
        if alignment is not None:
            layout = Layout[Byte](count=size, alignment=alignment.value())
        else:
            # use the default alignment
            layout = Layout[Byte](count=size)

        self._alloc = alloc(layout)

    def __deinit__(deinit self):
        dealloc(self._alloc^)

    def __init__(out self, *, copy: Self):
        self._alloc = alloc(copy._alloc.layout())
        self.span().copy_from(copy.span())

    @inline(.always)
    def length(self) -> Int:
        return self._alloc.layout().count()

    @inline(.always)
    def alignment(self) -> Int:
        return self._alloc.layout().alignment()

    @inline(.always)
    def unsafe_ptr(ref self) -> Pointer[Byte,origin_of(self._alloc._alloc)]:
        return self._alloc.unsafe_ptr()

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
    def span[origin: Origin](ref [origin] self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte, origin_of(self._alloc._alloc)]:

        var span_len = length.or_else(self.length() - start)

        # safety first
        debug_assert(
            start >= 0 and span_len >= 0 and start + span_len <= self.length(),
            "Invalid span: start=", start, ", length=", span_len, ", buf length=", self.length()
        )

        return self._alloc.unsafe_span()
            .unsafe_subspan(
                offset=start,
                length=span_len
            )
