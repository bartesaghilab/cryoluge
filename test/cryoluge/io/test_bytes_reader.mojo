
from testing import assert_equal, assert_raises

from cryoluge.io import ByteBuffer, BytesReader
from cryoluge_testlib import assert_equal_buffers


def test_init():
    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)
    assert_equal(reader.bytes_read(), 0)
    assert_equal(reader.bytes_remaining(), 3)


def test_read_bytes_once():
    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)
    var read_buf = InlineArray[Byte, 1](fill=0)
    assert_equal(reader.read_bytes(read_buf), 1)
    assert_equal_buffers(read_buf, InlineArray[Byte, 1](1))
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)


def test_read_bytes_multiple():

    var buf = InlineArray[Byte, 5](1, 2, 3, 4, 5)
    var reader = BytesReader(buf)

    var read1 = InlineArray[Byte, 2](fill=0)
    assert_equal(reader.read_bytes(read1), 2)
    assert_equal_buffers(read1, InlineArray[Byte, 2](1, 2))
    assert_equal(reader.bytes_read(), 2)
    assert_equal(reader.bytes_remaining(), 3)

    var read2 = InlineArray[Byte, 1](fill=0)
    assert_equal(reader.read_bytes(read2), 1)
    assert_equal_buffers(read2, InlineArray[Byte, 1](3))
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 2)

    var read3 = InlineArray[Byte, 2](fill=0)
    assert_equal(reader.read_bytes(read3), 2)
    assert_equal_buffers(read3, InlineArray[Byte, 2](4, 5))
    assert_equal(reader.bytes_read(), 5)
    assert_equal(reader.bytes_remaining(), 0)


def test_read_bytes_overflow():

    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)

    var read1 = InlineArray[Byte, 4](fill=0)
    with assert_raises():
        _ = reader.read_bytes(read1)

    var read2 = InlineArray[Byte, 3](fill=0)
    assert_equal(reader.read_bytes(read2), 3)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)

    var read3 = InlineArray[Byte, 1](fill=0)
    with assert_raises():
        _ = reader.read_bytes(read3)


def test_read_from_buffer():

    var buf = ByteBuffer(3)
    buf._p[0] = 1
    buf._p[1] = 2
    buf._p[2] = 3
    assert_equal_buffers(buf.span(), InlineArray[Byte, 3](1, 2, 3))

    var reader = BytesReader(buf.span())

    var read1 = InlineArray[Byte, 2](fill=0)
    assert_equal(reader.read_bytes(read1), 2)
    assert_equal_buffers(read1, InlineArray[Byte, 2](1, 2))
    assert_equal(reader.bytes_read(), 2)
    assert_equal(reader.bytes_remaining(), 1)

    assert_equal_buffers(buf.span(), InlineArray[Byte, 3](1, 2, 3))


def test_skip_bytes():

    var buf = InlineArray[Byte, 3](fill=0)
    var reader = BytesReader(buf)

    reader.skip_bytes(1)
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)

    with assert_raises():
        reader.skip_bytes(3)
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)

    reader.skip_bytes(2)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)

    reader.skip_bytes(0)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)

    with assert_raises():
        reader.skip_bytes(1)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)
    

def test_skip_scalar():

    var buf = InlineArray[Byte, 3](fill=0)
    var reader = BytesReader(buf)

    reader.skip_scalar[DType.int8]()
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)

    with assert_raises():
        reader.skip_scalar[DType.int32]()
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)

    reader.skip_scalar[DType.int16]()
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)

    with assert_raises():
        reader.skip_scalar[DType.int8]()
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)


def test_seek_to():

    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)

    reader.seek_to(0)
    assert_equal(reader.bytes_read(), 0)
    assert_equal(reader.bytes_remaining(), 3)
    assert_equal(reader.read_scalar[DType.uint8](), 1)

    reader.seek_to(1)
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)
    assert_equal(reader.read_scalar[DType.uint8](), 2)

    reader.seek_to(2)
    assert_equal(reader.bytes_read(), 2)
    assert_equal(reader.bytes_remaining(), 1)
    assert_equal(reader.read_scalar[DType.uint8](), 3)

    reader.seek_to(3)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)

    with assert_raises():
        reader.seek_to(4)


def test_seek_by():

    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)

    reader.seek_to(0)
    reader.seek_by(0)
    assert_equal(reader.bytes_read(), 0)
    assert_equal(reader.bytes_remaining(), 3)
    assert_equal(reader.read_scalar[DType.uint8](), 1)

    reader.seek_to(0)
    reader.seek_by(1)
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)
    assert_equal(reader.read_scalar[DType.uint8](), 2)

    reader.seek_to(0)
    reader.seek_by(2)
    assert_equal(reader.bytes_read(), 2)
    assert_equal(reader.bytes_remaining(), 1)
    assert_equal(reader.read_scalar[DType.uint8](), 3)

    reader.seek_to(0)
    reader.seek_by(3)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)
    with assert_raises():
        _ = reader.read_scalar[DType.uint8]()

    reader.seek_to(0)
    with assert_raises():
        reader.seek_by(4)

    reader.seek_to(2)
    reader.seek_by(0)
    assert_equal(reader.bytes_read(), 2)
    assert_equal(reader.bytes_remaining(), 1)
    assert_equal(reader.read_scalar[DType.uint8](), 3)

    reader.seek_to(2)
    reader.seek_by(1)
    assert_equal(reader.bytes_read(), 3)
    assert_equal(reader.bytes_remaining(), 0)
    with assert_raises():
        _ = reader.read_scalar[DType.uint8]()

    reader.seek_to(2)
    reader.seek_by(-1)
    assert_equal(reader.bytes_read(), 1)
    assert_equal(reader.bytes_remaining(), 2)
    assert_equal(reader.read_scalar[DType.uint8](), 2)

    reader.seek_to(2)
    reader.seek_by(-2)
    assert_equal(reader.bytes_read(), 0)
    assert_equal(reader.bytes_remaining(), 3)
    assert_equal(reader.read_scalar[DType.uint8](), 1)

    reader.seek_to(2)
    with assert_raises():
        reader.seek_by(-3)


def test_offset():

    var buf = InlineArray[Byte, 3](1, 2, 3)
    var reader = BytesReader(buf)

    assert_equal(reader.offset(), 0)

    reader.seek_to(0)
    assert_equal(reader.offset(), 0)

    reader.seek_to(3)
    assert_equal(reader.offset(), 3)

    reader.seek_by(-2)
    assert_equal(reader.offset(), 1)

    assert_equal(reader.read_scalar[DType.uint8](), 2)
    assert_equal(reader.offset(), 2)
