
from io import FileDescriptor


struct FileWriter(BinaryWriter, Movable):
    var _fd: FileDescriptor
    var _buf: ByteBuffer
    # TODO: how to write a self-referential struct?
    #var _buf_writer: BytesWriter[origin=__origin_of(self._buf)]
    # until we figure that out, just keep the pos in this struct
    var _pos: UInt

    def __init__(out self, fh: FileHandle, *, buf_size: UInt = 16*1024):
        self._fd = FileDescriptor(fh._get_raw_fd())
        # NOTE: _get_raw_fd() is an internal function, and therefore probably unstable?
        self._buf = ByteBuffer(buf_size)
        # TODO: how to do self references?
        #self._buf_writer = BytesWriter(self._buf.span())
        self._pos = 0

    fn write_bytes(mut self, bytes: Span[Byte]):

        var span_left = bytes
        var writer = BytesWriter(self._buf.span(), self._pos)

        while True:

            # how much is left to write?
            var bytes_left: UInt = len(span_left)
            if bytes_left == 0:
                break

            # write whatever will fit into the buffer
            var bytes_write = min(bytes_left, writer.bytes_remaining())
            writer.write_bytes(Span(span_left.unsafe_ptr(), bytes_write))
            span_left = Span(span_left.unsafe_ptr() + bytes_write, bytes_left - bytes_write)

            # flush the buffer, if needed
            if writer.bytes_remaining() == 0:
                self._fd.write_bytes(writer.span_written())
                writer.reset()
        
        self._pos = writer._pos

    fn write_scalar[dtype: DType](mut self, v: Scalar[dtype]):

        var writer = BytesWriter(self._buf.span(), self._pos)
        
        # make sure the scalar can fit in the buffer, even when empty
        var size = dtype.size_of()
        debug_assert[assert_mode="safe"](
             size <= self._buf.size(),
             "Buffer too small (", self._buf.size(), " bytes) to write scalar of ", size, " bytes"
        )

        # if the scalar won't fit in the remaining space, flush the buffer
        if size > writer.bytes_remaining():
            self._fd.write_bytes(writer.span_written())
            writer.reset()

        writer.write_scalar(v)

        # flush the buffer, if needed
        if writer.bytes_remaining() == 0:
            self._fd.write_bytes(writer.span_written())
            writer.reset()

        self._pos = writer._pos

    fn flush(mut self):
        var writer = BytesWriter(self._buf.span(), self._pos)
        self._fd.write_bytes(writer.span_written())
        writer.reset()
        self._pos = writer._pos
