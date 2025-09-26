
from memory import memcpy


struct BytesWriter[
    origin: Origin[mut=True]
](BinaryWriter, Movable):
    var buf: Span[Byte, origin].Mutable
    var _pos: UInt

    fn __init__(out self, buf: Span[Byte, origin].Mutable, pos: UInt = 0):
        self.buf = buf
        self._pos = pos

    fn bytes_written(self) -> UInt:
        return self._pos

    fn bytes_remaining(self) -> UInt:
        var capacity = len(self.buf)
        debug_assert(
            self._pos <= capacity,
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", capacity, ")"
        )
        return capacity - self._pos
        
    fn span_written(self) -> Span[Byte, origin]:
        return Span[Byte, origin](
            self.buf.unsafe_ptr(),
            self._pos
        )

    fn reset(mut self):
        self._pos = 0

    fn write_bytes(mut self, bytes: Span[Byte]):
        var size = len(bytes)
        debug_assert[assert_mode="safe"](
            size <= self.bytes_remaining(),
            "Buffer overflow: write=", size, ", remaining=", self.bytes_remaining()
        )
        var dst: UnsafePointer[Byte] = self.buf.unsafe_ptr() + self._pos
        var src: UnsafePointer[Byte] = bytes.unsafe_ptr()
        memcpy(dst, src, size)
        self._pos += size

    fn write_scalar[dtype: DType](mut self, v: Scalar[dtype]):
        var size = dtype.size_of()
        debug_assert[assert_mode="safe"](
            size <= self.bytes_remaining(),
            "Buffer overflow: write=", size, ", remaining=", self.bytes_remaining()
        )
        var dst = (self.buf.unsafe_ptr() + self._pos)
            .bitcast[Scalar[dtype]]()
        dst[] = v
        self._pos += size
