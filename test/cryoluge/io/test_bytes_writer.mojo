
from testing import assert_equal

from cryoluge.io import ByteBuffer, BytesWriter
from cryoluge_testlib import assert_equal_buffers


def test_init():
    var buf = InlineArray[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    assert_equal(writer.bytes_written(), 0)
    assert_equal(writer.bytes_remaining(), 3)
    assert_equal_buffers(buf, InlineArray[Byte, 3](0, 0, 0))


def test_write_bytes_once():
    var buf = InlineArray[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes(InlineArray[Byte, 2](1, 2))
    assert_equal(writer.bytes_written(), 2)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_buffers(buf, InlineArray[Byte, 3](1, 2, 0))


def test_write_bytes_multiple():
    var buf = InlineArray[Byte, 6](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes(InlineArray[Byte, 2](1, 2))
    writer.write_bytes(InlineArray[Byte, 1](3))
    writer.write_bytes(InlineArray[Byte, 2](4, 5))
    assert_equal(writer.bytes_written(), 5)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_buffers(buf, InlineArray[Byte, 6](1, 2, 3, 4, 5, 0))


def test_write_to_buffer():
    var buf = ByteBuffer(3)
    var writer = BytesWriter(buf.span())
    writer.write_bytes(InlineArray[Byte, 2](1, 2))
    assert_equal_buffers(buf.span(length=2), InlineArray[Byte, 2](1, 2))


def test_write_scalar():
    var buf = InlineArray[Byte, 4](fill=0)
    var writer = BytesWriter(buf)
    writer.write_scalar(UInt16(5))
    assert_equal(writer.bytes_written(), 2)
    assert_equal(writer.bytes_remaining(), 2)
    writer.write_scalar(UInt16(42))
    assert_equal(writer.bytes_written(), 4)
    assert_equal(writer.bytes_remaining(), 0)
    assert_equal_buffers(buf, InlineArray[Byte, 4](5, 0, 42, 0))
