
from testing import assert_equal

from cryoluge.io import ByteBuffer, BytesWriter, BinaryDataWriter
from cryoluge_testlib import assert_equal_buffers


def test_write_int():

    var buf = ByteBuffer(8)
    var writer = BytesWriter(buf.span())
    var data_writer = BinaryDataWriter(writer)

    data_writer.write_int(UInt8(5))
    assert_equal(writer.bytes_written(), 1)
    assert_equal_buffers(buf.span(length=1), InlineArray[Byte, 1](5))

    writer.reset()

    data_writer.write_int_be(Int64(578437695752307201))
    assert_equal(writer.bytes_written(), 8)
    assert_equal_buffers(buf.span(), InlineArray[Byte, 8](8, 7, 6, 5, 4, 3, 2, 1))

    writer.reset()

    data_writer.write_int_le(Int64(578437695752307201))
    assert_equal(writer.bytes_written(), 8)
    assert_equal_buffers(buf.span(), InlineArray[Byte, 8](1, 2, 3, 4, 5, 6, 7, 8))

    writer.reset()

    data_writer.write_int_be(Int64(-8608764254683430271))
    assert_equal_buffers(buf.span(), InlineArray[Byte, 8](136, 135, 134, 133, 132, 131, 130, 129))

    writer.reset()

    data_writer.write_int_le(Int64(-8608764254683430271))
    assert_equal_buffers(buf.span(), InlineArray[Byte, 8](129, 130, 131, 132, 133, 134, 135, 136))


def test_write_float():

    var buf = ByteBuffer(8)
    var writer = BytesWriter(buf.span())
    var data_writer = BinaryDataWriter(writer)

    data_writer.write_float(Float32(5.42))
    assert_equal(writer.bytes_written(), 4)

    writer.reset()

    data_writer.write_float(Float64(5.42))
    assert_equal(writer.bytes_written(), 8)
