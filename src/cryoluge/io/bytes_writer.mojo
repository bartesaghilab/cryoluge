
from sys import size_of

from memory import memcpy


struct BytesWriter[
    origin: Origin[mut=True]
](BinaryWriter, Movable):
    var buf: Span[Byte, origin].Mutable
    var _pos: Int

    def __init__(out self, buf: Span[Byte, origin].Mutable, pos: Int = 0):
        self.buf = buf
        self._pos = pos

    def bytes_written(self) -> Int:
        return self._pos

    def bytes_remaining(self) -> Int:
        var capacity = len(self.buf)
        debug_assert(
            self._pos <= capacity,
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", capacity, ")"
        )
        return capacity - self._pos
        
    def span_written(self) -> Span[Byte, origin]:
        return Span[Byte, origin](
            ptr=self.buf.unsafe_ptr(),
            length=self._pos
        )

    def reset(mut self):
        self._pos = 0

    def write_bytes(mut self, bytes: Span[Byte]):
        var size = len(bytes)
        debug_assert[assert_mode="safe"](
            size <= self.bytes_remaining(),
            "Buffer overflow: write=", size, ", remaining=", self.bytes_remaining()
        )
        var dst = self.buf.unsafe_ptr() + self._pos
        var src = bytes.unsafe_ptr()
        memcpy(src=src, dest=dst, count=size)
        self._pos += size

    def write_scalar[dtype: DType](mut self, v: Scalar[dtype]):
        var size = size_of[dtype]()
        debug_assert[assert_mode="safe"](
            size <= self.bytes_remaining(),
            "Buffer overflow: write=", size, ", remaining=", self.bytes_remaining()
        )
        var dst = (self.buf.unsafe_ptr() + self._pos)
            .bitcast[Scalar[dtype]]()
        dst[] = v
        self._pos += size
