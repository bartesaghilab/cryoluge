
struct BytesReader[
    origin: Origin[mut=False]
](BinaryReader, Movable):

    var buf: Span[Byte, origin]
    var _pos: Int

    fn __init__(out self, buf: Span[Byte, origin], pos: Int = 0):
        self.buf = buf
        self._pos = pos

    fn capacity(self) -> Int:
        return len(self.buf)

    fn bytes_read(self) -> Int:
        return self._pos

    fn bytes_remaining(self) -> Int:
        debug_assert(
            self._pos <= self.capacity(),
            "pos overflowed the buffer (pos=", self._pos, ", capacity=", self.capacity(), ")"
        )
        return self.capacity() - self._pos

    fn reset(mut self):
        self._pos = 0
        
    fn read_bytes(mut self, bytes: Span[mut=True, Byte]) raises -> Int:
        self.read_bytes_exact(bytes)
        return len(bytes)

    fn read_bytes_exact(mut self, bytes: Span[mut=True, Byte]) raises:
        var size = len(bytes)
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        var dst = bytes.unsafe_ptr()
        var src = self.buf.unsafe_ptr() + self._pos
        memcpy(src=src, dest=dst, count=size)
        self._pos += size

    fn read_scalar[dtype: DType](mut self, out v: Scalar[dtype]) raises:
        var size = size_of[dtype]()
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        var src = (self.buf.unsafe_ptr() + self._pos)
            .bitcast[Scalar[dtype]]()
        v = src[]
        self._pos += size

    fn skip_bytes(mut self, size: Int) raises:
        if size > self.bytes_remaining():
            raise Error(String("Buffer underflow: skip=", size, ", remaining=", self.bytes_remaining()))
        self._pos += size

    fn skip_scalar[dtype: DType](mut self) raises:
        self.skip_bytes(size_of[dtype]())

    fn offset(self) -> UInt64:
        return UInt64(self._pos)

    fn seek_to(mut self, offset: UInt64) raises:
        if offset > self.capacity():
            raise Error(String("Seek overflow: seek=", offset, ", capacity=", self.capacity()))
        self._pos = Int(offset)

    fn seek_by(mut self, offset: Int64) raises:
        var pos = self._pos + Int(offset)
        if pos < 0 or pos > self.capacity():
            raise Error(String("Seek overflow: offset_before=", self._pos, ", seek=", offset, ", offset_after=", pos))
        self._pos = pos
    