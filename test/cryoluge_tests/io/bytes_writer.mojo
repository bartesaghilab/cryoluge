
from std.testing import assert_equal

from cryoluge.io import ByteBuffer, BytesWriter
from cryoluge.test import assert_equal_buffers

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_init() raises:
    var buf = Array[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    assert_equal(writer.bytes_written(), 0)
    assert_equal(writer.bytes_remaining(), 3)
    assert_equal_buffers(buf, [0, 0, 0])


def test_write_bytes_once() raises:
    var buf = Array[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes([1, 2])
    assert_equal(writer.bytes_written(), 2)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_buffers(buf, [1, 2, 0])


def test_write_bytes_multiple() raises:
    var buf = Array[Byte, 6](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes([1, 2])
    writer.write_bytes([3])
    writer.write_bytes([4, 5])
    assert_equal(writer.bytes_written(), 5)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_buffers(buf, [1, 2, 3, 4, 5, 0])


def test_write_to_buffer() raises:
    var buf = ByteBuffer(3)
    var writer = BytesWriter(buf.span())
    writer.write_bytes([1, 2])
    assert_equal_buffers(buf.span(length=2), [1, 2])


def test_write_scalar() raises:
    var buf = Array[Byte, 4](fill=0)
    var writer = BytesWriter(buf)
    writer.write_scalar(UInt16(5))
    assert_equal(writer.bytes_written(), 2)
    assert_equal(writer.bytes_remaining(), 2)
    writer.write_scalar(UInt16(42))
    assert_equal(writer.bytes_written(), 4)
    assert_equal(writer.bytes_remaining(), 0)
    assert_equal_buffers(buf, [5, 0, 42, 0])
