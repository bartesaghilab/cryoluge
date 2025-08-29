
from io import FileHandle
from tempfile import NamedTemporaryFile

from testing import assert_equal

from cryoluge.io import FileWriter


def _handle(tempfile: NamedTemporaryFile) -> ref [tempfile._file_handle] FileHandle:
    return tempfile._file_handle
    # NOTE: _file_handle is internal, and therefore probably unstable?


def test_write_scalar_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(_handle(f))
        writer.write_scalar(UInt8(5))
        writer.flush()
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5])


def test_write_scalar_auto_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(_handle(f), 4)
        writer.write_scalar(UInt16(5))
        writer.write_scalar(UInt16(42))
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5, 0, 42, 0])


def test_write_scalar_auto_flush_multiple():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(_handle(f), 4)
        writer.write_scalar(UInt16(5))
        writer.write_scalar(UInt32(42))
        writer.write_scalar(UInt16(7))
        writer.write_scalar(UInt16(9))
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5, 0, 42, 0, 0, 0, 7, 0, 9, 0])
