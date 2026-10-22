
from std.io import RawFileDescriptor
from std.ffi import c_int

from cryoluge.io.buffers import ByteBuffer
from cryoluge.io.binary_writer import BinaryWriter
from cryoluge.io.bytes_writer import OwnedBytesWriter
from cryoluge.io.bytes_reader import BytesReader


struct FileWriter(
    BinaryWriter,
    Movable
):
    var _fd: RawFileDescriptor
    var _buf: OwnedBytesWriter

    def __init__(out self, fh: FileHandle, *, buf_size: Int = 16*1024) raises:
        self._fd = RawFileDescriptor(c_int(fh.handle))
        # NOTE: _get_raw_fd() is an internal function, and therefore probably unstable?
        self._buf = OwnedBytesWriter(ByteBuffer(buf_size))

    def __deinit__(deinit self):
        # flush any remaining buffers
        self.flush()

    def write_bytes(mut self, bytes: Span[Byte,_]):

        var reader = BytesReader(bytes)

        while True:

            if reader.bytes_remaining() <= 0:
                break

            # write whatever will fit into the buffer
            var size = min(reader.bytes_remaining(), self._buf.bytes_remaining())
            try:
                self._buf.write_from_reader(reader, size)
            except:
                pass  # reading from a byte buffer within its bounds *should be* infallible

            # flush the buffer, if needed
            if self._buf.bytes_remaining() == 0:
                self.flush()

    def flush(mut self):
        self._fd.write_bytes(self._buf.span_written())
        # amazingly, a file write operation doesn't raise, or return an error code
        self._buf.reset()
