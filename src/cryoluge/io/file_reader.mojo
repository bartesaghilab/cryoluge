
from io import FileDescriptor
from os import SEEK_CUR


struct FileReader[
    mut: Bool, //,
    origin: Origin[mut]
](BinaryReader):
    var _fh: Pointer[FileHandle, origin]
    var _fd: FileDescriptor
    var _buf: ByteBuffer
    var _pos: UInt
    var _limit: UInt

    def __init__(out self, ref [origin] fh: FileHandle, *, buf_size: UInt = 16*1024):
        self._fh = Pointer(to=fh)
        self._fd = FileDescriptor(fh._get_raw_fd())
        # NOTE: _get_raw_fd() is an internal function, and therefore probably unstable?
        self._buf = ByteBuffer(buf_size)
        self._pos = 0
        self._limit = 0

    fn read_bytes(mut self, buf: Span[mut=True, Byte]) raises -> UInt:

        var size = UInt(len(buf))
        var span_out = Span(buf.unsafe_ptr(), size)

        # read whatever's already in the buffer
        var size_buf_remaining = self._limit - self._pos
        if size_buf_remaining > 0:

            if size <= size_buf_remaining:
                # the buffer already has all we need: read all of it
                memcpy(
                    dest=span_out.unsafe_ptr(),
                    src=self._buf._p + self._pos,
                    count=size
                )
                self._pos += size
                return size

            # otherwise, read what we can from the buffer
            memcpy(
                dest=span_out.unsafe_ptr(),
                src=self._buf._p + self._pos,
                count=size_buf_remaining
            )
            span_out = Span(buf.unsafe_ptr() + size_buf_remaining, size - size_buf_remaining)

        # need more: read from the fd into the buffer
        var size_read = self._fd.read_bytes(self._buf.span())
        if size_read == 0:
            return size_buf_remaining
        self._pos = 0
        self._limit = size_read

        # then copy whatever else we can from the buffer
        var size_copy = min(size_read, UInt(len(span_out)))
        memcpy(
            dest=span_out.unsafe_ptr(),
            src=self._buf._p,
            count=size_copy
        )
        self._pos += size_copy
        return size_buf_remaining + size_copy

    fn read_bytes_exact(mut self, buf: Span[mut=True, Byte]) raises:

        var size = UInt(len(buf))
        var size_out_read = UInt(0)

        while size_out_read < size:
            var span_out = Span(buf.unsafe_ptr() + size_out_read, size - size_out_read)
            var size_read = self.read_bytes(span_out)
            if size_read == 0:
                raise Error(String("Can't read ", size, " bytes: EOF"))
            size_out_read += size_read

        # just in case ...
        debug_assert(
            size_out_read == size,
            "Failed to read exact buffer size: read=", size_out_read, ", buf=", size
        )

    fn read_scalar[dtype: DType](mut self, out v: Scalar[dtype]) raises:
        
        # make sure the scalar can fit in the buffer, even when empty
        var size = dtype.sizeof()
        debug_assert[assert_mode="safe"](
             size <= self._buf.size(),
             "Buffer too small (", self._buf.size(), " bytes) to read scalar of ", size, " bytes"
        )

        var size_buf_remaining = self._limit - self._pos
        
        if size_buf_remaining < size:

            if size_buf_remaining > 0:
                # copy leftovers to start of buffer
                memcpy(
                    dest=self._buf._p,
                    src=self._buf._p + self._pos,
                    count=size_buf_remaining
                )

            # read into the buffer
            var span_read = self._buf.span(
                start=size_buf_remaining,
                length=Int(self._buf.size() - size_buf_remaining)
            )
            var size_read = self._fd.read_bytes(span_read)
            if size_read == 0:
                raise Error(String("Can't read ", dtype, ": EOF"))

            self._pos = 0
            self._limit = size_buf_remaining + size_read

        # finally, value is in-memory: read it
        var reader = BytesReader(self._buf.span(start=self._pos, length=size), 0)
        v = reader.read_scalar[dtype]()
        self._pos += reader._pos

    fn skip_bytes(mut self, size: UInt) raises:

        # if we've already buffered the skip size, just advance the buffer position
        var size_buf_remaining = self._limit - self._pos
        if size <= size_buf_remaining:
            self._pos += size
            return

        # otherwise, ignore the buffer
        self._pos = self._limit

        # and seek ahead, if needed
        var size_seek = size - size_buf_remaining
        if size_seek > 0:
            _ = self._fh[].seek(size_seek, SEEK_CUR)

    fn skip_scalar[dtype: DType](mut self) raises:
        self.skip_bytes(dtype.sizeof())
