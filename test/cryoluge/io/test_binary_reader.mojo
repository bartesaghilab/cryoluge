
from testing import assert_equal, assert_raises
from memory import memcpy

from cryoluge.io import ByteBuffer, BytesReader, Endian
from cryoluge_testlib import assert_equal_buffers


def write_buf(mut buf: ByteBuffer, bytes: Span[Byte]):
    var src = bytes
    var size = len(src)
    var dst = buf.span(length=size)
    debug_assert(
        size <= len(dst),
        "Buffer overflow: write=", size, ", capacity=", len(dst)
    )
    dst.copy_from(src)


def test_read_scalars():

    var buf = ByteBuffer(8)
    var reader = BytesReader(buf.span())

    # 1-byte integers
    write_buf(buf, InlineArray[Byte, 1](5))
    assert_equal(reader.read_u8(), UInt8(5))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 1](251))
    assert_equal(reader.read_i8(), Int8(-5))

    reader.reset()

    # 2-byte integers
    write_buf(buf, InlineArray[Byte, 2](1, 2))
    assert_equal(reader.read_u16[Endian.Little](), UInt16(513))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 2](2, 1))
    assert_equal(reader.read_u16[Endian.Big](), UInt16(513))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 2](255, 253))
    assert_equal(reader.read_i16[Endian.Little](), Int16(-513))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 2](253, 255))
    assert_equal(reader.read_i16[Endian.Big](), Int16(-513))

    reader.reset()

    # 4-byte integers
    write_buf(buf, InlineArray[Byte, 4](1, 2, 3, 4))
    assert_equal(reader.read_u32[Endian.Little](), UInt32(67305985))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 4](4, 3, 2, 1))
    assert_equal(reader.read_u32[Endian.Big](), UInt32(67305985))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 4](255, 253, 252, 251))
    assert_equal(reader.read_i32[Endian.Little](), Int32(-67305985))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 4](251, 252, 253, 255))
    assert_equal(reader.read_i32[Endian.Big](), Int32(-67305985))

    reader.reset()

    # 8-byte integers
    write_buf(buf, InlineArray[Byte, 8](1, 2, 3, 4, 5, 6, 7, 8))
    assert_equal(reader.read_u64[Endian.Little](), UInt64(578437695752307201))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 8](8, 7, 6, 5, 4, 3, 2, 1))
    assert_equal(reader.read_u64[Endian.Big](), UInt64(578437695752307201))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 8](255, 253, 252, 251, 250, 249, 248, 247))
    assert_equal(reader.read_i64[Endian.Little](), Int64(-578437695752307201))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 8](247, 248, 249, 250, 251, 252, 253, 255))
    assert_equal(reader.read_i64[Endian.Big](), Int64(-578437695752307201))

    reader.reset()

    # 4-byte floats
    write_buf(buf, InlineArray[Byte, 4](0b10110000, 0b11000001, 0b01001110, 0b01001100))
    assert_equal(reader.read_f32[Endian.Little](), Float32(5.42e7))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 4](0b01001100, 0b01001110, 0b11000001, 0b10110000))
    assert_equal(reader.read_f32[Endian.Big](), Float32(5.42e7))

    reader.reset()

    # 8-byte floats
    write_buf(buf, InlineArray[Byte, 8](
        0b01011000, 0b11111010, 0b01001000, 0b00110001,
        0b00101010, 0b11101110, 0b01000101, 0b01000011
    ))
    assert_equal(reader.read_f64[Endian.Little](), Float64(1.2345678987654321e16))
    reader.reset()
    write_buf(buf, InlineArray[Byte, 8](
        0b01000011, 0b01000101, 0b11101110, 0b00101010,
        0b00110001, 0b01001000, 0b11111010, 0b01011000
    ))
    assert_equal(reader.read_f64[Endian.Big](), Float64(1.2345678987654321e16))

