
struct BytesReader[
    origin: Origin[mut=False], //
](BinaryReader):

    var buf: Span[Byte, origin]
    var _pos: UInt

    fn __init__(out self, buf: Span[Byte, origin]):
        self.buf = buf
        self._pos = 0

    fn bytes_read(self) -> UInt:
        return self._pos

    fn bytes_remaining(self) -> UInt:
        var capacity = len(self.buf)
        debug_assert(
            self._pos <= capacity,
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", capacity, ")"
        )
        return capacity - self._pos

    fn reset(mut self):
        self._pos = 0
        
    fn read_bytes(mut self, bytes: MutByteSpan) raises -> UInt:
        var size = len(bytes)
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        var dst: UnsafePointer[Byte] = bytes.unsafe_ptr()
        var src: UnsafePointer[Byte] = self.buf.unsafe_ptr() + self._pos
        memcpy(dst, src, size)
        self._pos += size
        return size
