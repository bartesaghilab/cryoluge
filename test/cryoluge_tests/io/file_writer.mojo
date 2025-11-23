
from io import FileHandle
from tempfile import NamedTemporaryFile

from testing import assert_equal

from cryoluge.io import FileWriter

from cryoluge_testlib import file_handle


comptime funcs = __functions_in_module()


def test_write_bytes_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(file_handle(f))
        writer.write_bytes(InlineArray[Byte, 4](1, 2, 3, 4))
        writer.flush()
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [1, 2, 3, 4])


def test_write_bytes_auto_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(file_handle(f), buf_size=2)
        writer.write_bytes(InlineArray[Byte, 6](1, 2, 3, 4, 5, 6))
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [1, 2, 3, 4, 5, 6])


def test_write_scalar_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(file_handle(f))
        writer.write_scalar(UInt8(5))
        writer.flush()
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5])


def test_write_scalars_auto_flush():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(file_handle(f), buf_size=4)
        writer.write_scalar(UInt16(5))
        writer.write_scalar(UInt16(42))
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5, 0, 42, 0])


def test_write_scalars_auto_flush_multiple():
    with NamedTemporaryFile(mode="rw") as f:
        var writer = FileWriter(file_handle(f), buf_size=4)
        writer.write_scalar(UInt16(5))
        writer.write_scalar(UInt32(42))
        writer.write_scalar(UInt16(7))
        writer.write_scalar(UInt16(9))
        _ = f.seek(0)
        assert_equal(f.read_bytes(), [5, 0, 42, 0, 0, 0, 7, 0, 9, 0])
