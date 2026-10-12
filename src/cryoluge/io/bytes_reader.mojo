
from cryoluge.io.binary_reader import BinaryReader


struct BytesReader[
    origin: Origin[mut=False]
](BinaryReader, Movable):

    var buf: Span[Byte, Self.origin]
    var _pos: Int

    @inline(.always)
    def __init__(out self, buf: Span[Byte, Self.origin], pos: Int = 0):
        self.buf = buf
        self._pos = pos

    @inline(.always)
    def capacity(self) -> Int:
        return len(self.buf)

    @inline(.always)
    def bytes_read(self) -> Int:
        return self._pos

    @inline(.always)
    def bytes_remaining(self) -> Int:
        debug_assert(
            self._pos <= self.capacity(),
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", self.capacity(), ")"
        )
        return self.capacity() - self._pos

    @inline(.always)
    def span_remaining(ref self, *, length: Optional[Int] = None) -> Span[Byte,Self.origin]:
        var bytes_remaining = self.bytes_remaining()
        if length is not None:
            debug_assert(
                length.value() <= bytes_remaining,
                t"Can't read {length.value()} bytes, only {bytes_remaining} remaining"
            )
        return self.buf.unsafe_subspan(
            offset=self._pos,
            length=length.or_else(bytes_remaining)
        )

    @inline(.always)
    def reset(mut self):
        self._pos = 0
        
    def read_bytes[origin_dst: MutOrigin](mut self, dst: Span[Byte,origin_dst]) raises -> Int:
        self.read_bytes_exact(dst)
        return len(dst)

    def read_bytes_exact[origin_dst: MutOrigin](mut self, dst: Span[Byte,origin_dst]) raises:
        var size = len(dst)
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        dst.copy_from(self.span_remaining(length=size))
        self._pos += size

    def skip_bytes(mut self, size: Int) raises:
        if size > self.bytes_remaining():
            raise Error(String("Buffer underflow: skip=", size, ", remaining=", self.bytes_remaining()))
        self._pos += size

    def offset(self) -> Int:
        return self._pos

    def seek_to(mut self, offset: Int) raises:
        if offset > self.capacity():
            raise Error(String("Seek overflow: seek=", offset, ", capacity=", self.capacity()))
        self._pos = offset

    def seek_by(mut self, offset: Int) raises:
        var pos = self._pos + offset
        if pos < 0 or pos > self.capacity():
            raise Error(String("Seek overflow: offset_before=", self._pos, ", seek=", offset, ", offset_after=", pos))
        self._pos = pos
    