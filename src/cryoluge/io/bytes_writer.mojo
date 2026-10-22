
from std.sys import size_of

from cryoluge.io import as_byte_span
from cryoluge.io.buffers import ByteBuffer
from cryoluge.io.binary_writer import BinaryWriter
from cryoluge.io.bytes_reader import BytesReader


struct BytesWriter[
    origin: MutOrigin
](BinaryWriter, Movable):
    """
    Writes bytes to a buffer held by reference.
    """

    var buf: Span[Byte, Self.origin]
    var _inner: _Inner

    def __init__(out self, buf: Span[Byte,Self.origin], pos: Int = 0):
        self.buf = buf
        self._inner = _Inner(pos)

    def bytes_written(self) -> Int:
        return self._inner.bytes_written()

    def bytes_remaining(self) -> Int:
        return self._inner.bytes_remaining(self.buf)
        
    def span_written(self) -> Span[Byte,Self.origin]:
        return self._inner.span_written(self.buf)

    def reset(mut self):
        self._inner.reset()

    def write_bytes(mut self, bytes: Span[Byte,_]):
        self._inner.write_bytes(self.buf, bytes)

    def write_from_reader(mut self, mut src: BytesReader, length: Int) raises:
        self._inner.write_from_reader(self.buf, src, length)


struct OwnedBytesWriter(BinaryWriter, Movable):
    """
    Writes bytes to a buffer given by value.
    """

    var _buf: ByteBuffer
    var _inner: _Inner

    def __init__(out self, var buf: ByteBuffer, pos: Int = 0):
        self._buf = buf^
        self._inner = _Inner(pos)

    def bytes_written(self) -> Int:
        return self._inner.bytes_written()

    def bytes_remaining(self) -> Int:
        return self._inner.bytes_remaining(self._buf.span())
        
    def span_written(self) -> Span[Byte,origin_of(self._buf)]:
        return self._inner.span_written(self._buf.span())

    def reset(mut self):
        self._inner.reset()

    def write_bytes(mut self, bytes: Span[Byte,_]):
        self._inner.write_bytes(self._buf.span(), bytes)

    def write_from_reader(mut self, mut src: BytesReader, length: Int) raises:
        self._inner.write_from_reader(self._buf.span(), src, length)


struct _Inner(Movable):
    """
    Self-referential structs are a huge pain in lifetime/origin-tracking languages.
    This version of `BytesWriter` works without storing the buffer reference as a struct member.
    """

    var _pos: Int

    @inline(.always)
    def __init__(out self, pos: Int = 0):
        self._pos = pos

    @inline(.always)
    def bytes_written(self) -> Int:
        return self._pos

    @inline(.always)
    def capacity(self, buf: Span[Byte,_]) -> Int:
        var capacity = len(buf)
        debug_assert(
            self._pos <= capacity,
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", capacity, ")"
        )
        return capacity

    @inline(.always)
    def bytes_remaining(self, buf: Span[Byte,_]) -> Int:
        return self.capacity(buf) - self._pos
        
    @inline(.always)
    def span_written[origin: Origin](self, buf: Span[Byte,origin]) -> Span[Byte,origin]:
        _ = self.capacity(buf)
        return buf.unsafe_subspan(offset=0, length=self._pos)

    @inline(.always)
    def reset(mut self):
        self._pos = 0

    def write_bytes[origin: MutOrigin](mut self, buf: Span[Byte,origin], bytes: Span[Byte,_]):
        var size = len(bytes)
        var bytes_remaining = self.bytes_remaining(buf)
        debug_assert(
            size <= bytes_remaining,
            "Buffer overflow: write=", size, ", remaining=", bytes_remaining
        )
        buf.unsafe_subspan(offset=self._pos, length=size)
            .copy_from(bytes)
        self._pos += size

    def write_from_reader[origin: MutOrigin](mut self, buf: Span[Byte,origin], mut src: BytesReader, length: Int) raises:
        self.write_bytes(buf, src.span_remaining(length=length))
        src.skip_bytes(length)
