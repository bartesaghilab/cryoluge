
struct BytesReader[
    origin: Origin[mut=False], //
](BinaryReader):

    var buf: Span[Byte, origin]
    var _pos: UInt

    fn __init__(out self, buf: Span[Byte, origin], pos: UInt = 0):
        self.buf = buf
        self._pos = pos

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
        
    fn read_bytes(mut self, bytes: Span[mut=True, Byte]) raises -> UInt:
        self.read_bytes_exact(bytes)
        return len(bytes)

    fn read_bytes_exact(mut self, bytes: Span[mut=True, Byte]) raises:
        var size = len(bytes)
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        var dst: UnsafePointer[Byte] = bytes.unsafe_ptr()
        var src: UnsafePointer[Byte] = self.buf.unsafe_ptr() + self._pos
        memcpy(dst, src, size)
        self._pos += size

    fn read_scalar[dtype: DType](mut self, out v: Scalar[dtype]) raises:
        var size = dtype.size_of()
        if size > self.bytes_remaining():
            raise Error("Buffer underflow: read=", size, ", remaining=", self.bytes_remaining())
        var src = (self.buf.unsafe_ptr() + self._pos)
            .bitcast[Scalar[dtype]]()
        v = src[]
        self._pos += size

    fn skip_bytes(mut self, size: UInt) raises:
        if size > self.bytes_remaining():
            raise Error(String("Buffer underflow: skip=", size, ", remaining=", self.bytes_remaining()))
        self._pos += size

    fn skip_scalar[dtype: DType](mut self) raises:
        self.skip_bytes(dtype.size_of())
