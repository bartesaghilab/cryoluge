
from std.io import RawFileDescriptor
from std.os import SEEK_CUR, SEEK_SET
from std.ffi import c_int

from cryoluge.io.buffers import ByteBuffer
from cryoluge.io.binary_reader import BinaryReader


comptime DEFAULT_BUFFER_SIZE = 16*1024


struct FileReader[origin: Origin](
    BinaryReader,
    Movable
):
    var _fh: Pointer[FileHandle, Self.origin]
    var _inner: _Inner

    def __init__(out self, ref [Self.origin] fh: FileHandle, *, buf_size: Int = DEFAULT_BUFFER_SIZE):
        self._fh = Pointer(to=fh)
        self._inner = _Inner(self._fh[], buf_size)

    def read_bytes[o: MutOrigin](mut self, buf: Span[Byte,o]) raises -> Int:
        return self._inner.read_bytes(buf)

    def read_bytes_exact[o: MutOrigin](mut self, buf: Span[Byte,o]) raises:
        return self._inner.read_bytes_exact(buf)
    
    def skip_bytes(mut self, size: Int) raises:
        return self._inner.skip_bytes(self._fh[], size)

    def offset(self) -> Int:
        return self._inner.offset()

    def seek_to(mut self, offset: Int) raises:
        self._inner.seek_to(self._fh[], offset)

    def seek_by(mut self, offset: Int) raises:
        self._inner.seek_by(self._fh[], offset)


struct OwnedFileReader(
    BinaryReader,
    Movable
):
    var _fh: FileHandle
    var _inner: _Inner

    def __init__(out self, path: String, *, buf_size: Int = DEFAULT_BUFFER_SIZE) raises:
        self._fh = open(path, 'r')
        self._inner = _Inner(self._fh, buf_size)

    def __init__(out self, var fh: FileHandle, *, buf_size: Int = DEFAULT_BUFFER_SIZE):
        self._fh = fh^
        self._inner = _Inner(self._fh, buf_size)

    def read_bytes[origin: MutOrigin](mut self, buf: Span[Byte,origin]) raises -> Int:
        return self._inner.read_bytes(buf)

    def read_bytes_exact[origin: MutOrigin](mut self, buf: Span[Byte,origin]) raises:
        return self._inner.read_bytes_exact(buf)
    
    def skip_bytes(mut self, size: Int) raises:
        return self._inner.skip_bytes(self._fh, size)

    def offset(self) -> Int:
        return self._inner.offset()

    def seek_to(mut self, offset: Int) raises:
        self._inner.seek_to(self._fh, offset)

    def seek_by(mut self, offset: Int) raises:
        self._inner.seek_by(self._fh, offset)


@fieldwise_init
struct _Inner(Movable):
    var _fd: RawFileDescriptor
    var _buf: ByteBuffer[]
    var _pos: Int
    var _limit: Int
    var _offset: Int

    def __init__(out self, fh: FileHandle, buf_size: Int):
        self._fd = RawFileDescriptor(c_int(fh.handle))
        # NOTE: _get_raw_fd() is an internal function, and therefore probably unstable?
        self._buf = ByteBuffer(buf_size)
        self._pos = 0
        self._limit = 0
        self._offset = 0

    def read_bytes[origin: MutOrigin](mut self, buf: Span[Byte,origin]) raises -> Int:

        var size = len(buf)
        var span_out = buf

        # read whatever's already in the buffer
        var size_buf_remaining = self._limit - self._pos
        if size_buf_remaining > 0:

            if size <= size_buf_remaining:
                # the buffer already has all we need: read all of it
                span_out.copy_from(self._buf.span(start=self._pos, length=size))
                self._pos += size
                self._offset += size
                return size

            # otherwise, read what we can from the buffer
            span_out
                .unsafe_subspan(offset=0, length=size_buf_remaining)
                .copy_from(self._buf.span(start=self._pos, length=size_buf_remaining))
            span_out = span_out.unsafe_subspan(
                offset=size_buf_remaining,
                length=size - size_buf_remaining
            )

        # need more: read from the fd into the buffer
        var size_read = self._fd.read_bytes(self._buf.span())
        if size_read <= 0:
            self._offset += size_buf_remaining
            return size_buf_remaining
        self._pos = 0
        self._limit = size_read

        # then copy whatever else we can from the buffer
        var size_copy = min(size_read, len(span_out))
        span_out
            .unsafe_subspan(offset=0, length=size_copy)
            .copy_from(self._buf.span(start=0, length=size_copy))
        self._pos += size_copy
        self._offset += size_buf_remaining + size_copy
        return size_buf_remaining + size_copy

    def read_bytes_exact[origin: MutOrigin](mut self, buf: Span[Byte,origin]) raises:

        var size = len(buf)
        var size_out_read = 0

        while size_out_read < size:
            var span_out = buf.unsafe_subspan(
                offset=size_out_read,
                length=size - size_out_read
            )
            var size_read = self.read_bytes(span_out)
            if size_read == 0:
                raise Error(String("Can't read ", size, " bytes: EOF"))
            size_out_read += size_read

        # just in case ...
        debug_assert(
            size_out_read == size,
            "Failed to read exact buffer size: read=", size_out_read, ", buf=", size
        )

    def skip_bytes(mut self, fh: FileHandle, size: Int) raises:

        # if we've already buffered the skip size, just advance the buffer position
        var size_buf_remaining = self._limit - self._pos
        if size <= size_buf_remaining:
            self._pos += size
            self._offset += size
            return

        # otherwise, ignore the buffer
        self._pos = self._limit

        # and seek ahead, if needed
        var size_seek = size - size_buf_remaining
        if size_seek > 0:
            _ = fh.seek(size_seek, SEEK_CUR)

        self._offset += size

    def offset(self) -> Int:
        return self._offset

    def seek_to(mut self, fh: FileHandle, offset: Int) raises:

        # ignore the current buffer
        self._pos = self._limit

        # seek the file
        _ = fh.seek(offset, SEEK_SET)

        self._offset = offset

    def seek_by(mut self, fh: FileHandle, offset: Int) raises:

        var new_offset = self._offset + offset
        if new_offset < 0:
            raise Error(String("Seek underflow: offset_before=", self._offset, ", seek=", offset, ", offset_after=", new_offset))

        # ignore the current buffer
        self._pos = self._limit

        # seek the file
        _ = fh.seek(offset, SEEK_CUR)
        
        self._offset = new_offset
