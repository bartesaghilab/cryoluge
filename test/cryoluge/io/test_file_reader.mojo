
from io import FileHandle
from tempfile import NamedTemporaryFile

from testing import assert_equal, assert_raises

from cryoluge.io import FileReader

from cryoluge_testlib import assert_equal_buffers, file_handle


def test_read_bytes():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 1](5))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f))
        var buf = InlineArray[Byte, 1](fill=0)
        assert_equal(reader.read_bytes(buf), 1)
        assert_equal_buffers(buf, InlineArray[Byte, 1](5))


def test_read_bytes_partial():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        var buf = InlineArray[Byte, 3](fill=0)
        assert_equal(reader.read_bytes(buf), 2)
        assert_equal_buffers(buf, InlineArray[Byte, 3](1, 2, 0))


def test_read_bytes_partial_leftovers():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 4](1, 2, 3, 4))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)

        var buf1 = InlineArray[Byte, 1](fill=0)
        assert_equal(reader.read_bytes(buf1), 1)
        assert_equal_buffers(buf1, InlineArray[Byte, 1](1))

        var buf2 = InlineArray[Byte, 3](fill=0)
        assert_equal(reader.read_bytes(buf2), 3)
        assert_equal_buffers(buf2, InlineArray[Byte, 3](2, 3, 4))


def test_read_bytes_eof():
    with NamedTemporaryFile(mode="rw") as f:
        var reader = FileReader(file_handle(f))

        var buf1 = InlineArray[Byte, 1](fill=0)
        assert_equal(reader.read_bytes(buf1), 0)
        assert_equal_buffers(buf1, InlineArray[Byte, 1](0))


def test_read_bytes_leftovers_eof():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)

        var buf1 = InlineArray[Byte, 1](fill=0)
        assert_equal(reader.read_bytes(buf1), 1)
        assert_equal_buffers(buf1, InlineArray[Byte, 1](1))

        var buf2 = InlineArray[Byte, 3](fill=0)
        assert_equal(reader.read_bytes(buf2), 2)
        assert_equal_buffers(buf2, InlineArray[Byte, 3](2, 3, 0))

        var buf3 = InlineArray[Byte, 1](fill=0)
        assert_equal(reader.read_bytes(buf3), 0)
        assert_equal_buffers(buf3, InlineArray[Byte, 1](0))


def test_read_bytes_exact():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 5](1, 2, 3, 4, 5))

        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        var buf1 = InlineArray[Byte, 1](fill=0)
        reader.read_bytes_exact(buf1)
        assert_equal_buffers(buf1, InlineArray[Byte, 1](1))

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        var buf2 = InlineArray[Byte, 2](fill=0)
        reader.read_bytes_exact(buf2)
        assert_equal_buffers(buf2, InlineArray[Byte, 2](1, 2))

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        var buf3 = InlineArray[Byte, 3](fill=0)
        reader.read_bytes_exact(buf3)
        assert_equal_buffers(buf3, InlineArray[Byte, 3](1, 2, 3))

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        var buf4 = InlineArray[Byte, 4](fill=0)
        reader.read_bytes_exact(buf4)
        assert_equal_buffers(buf4, InlineArray[Byte, 4](1, 2, 3, 4))

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        var buf5 = InlineArray[Byte, 5](fill=0)
        reader.read_bytes_exact(buf5)
        assert_equal_buffers(buf5, InlineArray[Byte, 5](1, 2, 3, 4, 5))

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        var buf6 = InlineArray[Byte, 6](fill=0)
        with assert_raises():
            reader.read_bytes_exact(buf6)
        assert_equal_buffers(buf6, InlineArray[Byte, 6](1, 2, 3, 4, 5, 0))


def test_read_scalar():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 1](5))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f))
        assert_equal(reader.read_scalar[DType.uint8](), UInt8(5))


def test_read_scalars():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 4](5, 0, 42, 0))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        assert_equal(reader.read_scalar[DType.uint16](), UInt16(5))
        assert_equal(reader.read_scalar[DType.uint16](), UInt16(42))


def test_read_scalars_leftover_copy():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 4](5, 42, 0, 7))
        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        assert_equal(reader.read_scalar[DType.uint8](), UInt8(5))
        assert_equal(reader.read_scalar[DType.uint16](), UInt16(42))
        assert_equal(reader.read_scalar[DType.uint8](), UInt8(7))


def test_skip_bytes():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 4](1, 2, 3, 4))

        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(1)
        assert_equal(reader.read_scalar[DType.uint8](), 2)

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(2)
        assert_equal(reader.read_scalar[DType.uint8](), 3)

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(3)
        assert_equal(reader.read_scalar[DType.uint8](), 4)

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(4)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(5)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()


def test_skip_bytes_offset():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 4](1, 2, 3, 4))

        _ = f.seek(0)
        var reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(1)
        reader.skip_bytes(1)
        assert_equal(reader.read_scalar[DType.uint8](), 3)

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(1)
        reader.skip_bytes(2)
        assert_equal(reader.read_scalar[DType.uint8](), 4)

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(1)
        reader.skip_bytes(3)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        _ = f.seek(0)
        reader = FileReader(file_handle(f), buf_size=2)
        reader.skip_bytes(1)
        reader.skip_bytes(4)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()


def test_skip_scalar():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))

        _ = f.seek(0)
        var reader = FileReader(file_handle(f))
        reader.skip_scalar[DType.int8]()
        assert_equal(reader.read_scalar[DType.uint8](), 2)


def test_seek_to():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))
        var reader = FileReader(file_handle(f))

        reader.seek_to(0)
        assert_equal(reader.read_scalar[DType.uint8](), 1)

        reader.seek_to(1)
        assert_equal(reader.read_scalar[DType.uint8](), 2)

        reader.seek_to(2)
        assert_equal(reader.read_scalar[DType.uint8](), 3)

        reader.seek_to(3)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        reader.seek_to(4)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()


def test_seek_by():
    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))
        var reader = FileReader(file_handle(f))

        reader.seek_to(0)
        reader.seek_by(0)
        assert_equal(reader.read_scalar[DType.uint8](), 1)

        reader.seek_to(0)
        reader.seek_by(1)
        assert_equal(reader.read_scalar[DType.uint8](), 2)

        reader.seek_to(0)
        reader.seek_by(2)
        assert_equal(reader.read_scalar[DType.uint8](), 3)

        reader.seek_to(0)
        reader.seek_by(3)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        reader.seek_to(0)
        reader.seek_by(4)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        reader.seek_to(2)
        reader.seek_by(0)
        assert_equal(reader.read_scalar[DType.uint8](), 3)

        reader.seek_to(2)
        reader.seek_by(1)
        with assert_raises():
            _ = reader.read_scalar[DType.uint8]()

        reader.seek_to(2)
        reader.seek_by(-1)
        assert_equal(reader.read_scalar[DType.uint8](), 2)

        reader.seek_to(2)
        reader.seek_by(-2)
        assert_equal(reader.read_scalar[DType.uint8](), 1)

        reader.seek_to(2)
        with assert_raises():
            reader.seek_by(-3)


def test_offset():

    with NamedTemporaryFile(mode="rw") as f:
        f.write_bytes(InlineArray[Byte, 3](1, 2, 3))
        var reader = FileReader(file_handle(f))
        _ = f.seek(0)
    
        assert_equal(reader.offset(), 0)

        reader.seek_to(0)
        assert_equal(reader.offset(), 0)

        reader.seek_to(3)
        assert_equal(reader.offset(), 3)

        reader.seek_by(-2)
        assert_equal(reader.offset(), 1)

        assert_equal(reader.read_scalar[DType.uint8](), 2)
        assert_equal(reader.offset(), 2)

        # TODO: test more code paths in read_bytes() ?
