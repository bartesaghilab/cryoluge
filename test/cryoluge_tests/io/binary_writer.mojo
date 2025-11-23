
from testing import assert_equal

from cryoluge.io import ByteBuffer, BytesWriter, Endian
from cryoluge_testlib import assert_equal_buffers


comptime funcs = __functions_in_module()


def test_write_scalars():

    var buf = ByteBuffer(8)
    var writer = BytesWriter(buf.span())

    # 1-byte integers
    writer.write_u8(5)
    assert_equal_buffers(buf.span(length=1), InlineArray[Byte, 1](5))
    writer.reset()
    writer.write_i8(-5)
    assert_equal_buffers(buf.span(length=1), InlineArray[Byte, 1](251))

    writer.reset()

    # 2-byte integers
    writer.write_u16[Endian.Little](513)
    assert_equal_buffers(buf.span(length=2), InlineArray[Byte, 2](1, 2))
    writer.reset()
    writer.write_u16[Endian.Big](513)
    assert_equal_buffers(buf.span(length=2), InlineArray[Byte, 2](2, 1))
    writer.reset()
    writer.write_i16[Endian.Little](-513)
    assert_equal_buffers(buf.span(length=2), InlineArray[Byte, 2](255, 253))
    writer.reset()
    writer.write_i16[Endian.Big](-513)
    assert_equal_buffers(buf.span(length=2), InlineArray[Byte, 2](253, 255))

    writer.reset()

    # 4-byte integers
    writer.write_u32[Endian.Little](67305985)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](1, 2, 3, 4))
    writer.reset()
    writer.write_u32[Endian.Big](67305985)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](4, 3, 2, 1))
    writer.reset()
    writer.write_i32[Endian.Little](-67305985)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](255, 253, 252, 251))
    writer.reset()
    writer.write_i32[Endian.Big](-67305985)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](251, 252, 253, 255))

    writer.reset()

    # 8-byte integers
    writer.write_u64[Endian.Little](578437695752307201)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](1, 2, 3, 4, 5, 6, 7, 8))
    writer.reset()
    writer.write_u64[Endian.Big](578437695752307201)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](8, 7, 6, 5, 4, 3, 2, 1))
    writer.reset()
    writer.write_i64[Endian.Little](-578437695752307201)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](255, 253, 252, 251, 250, 249, 248, 247))
    writer.reset()
    writer.write_i64[Endian.Big](-578437695752307201)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](247, 248, 249, 250, 251, 252, 253, 255))

    writer.reset()

    # 4-byte floats
    writer.write_f32[Endian.Little](5.42e7)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](0b10110000, 0b11000001, 0b01001110, 0b01001100))
    writer.reset()
    writer.write_f32[Endian.Big](5.42e7)
    assert_equal_buffers(buf.span(length=4), InlineArray[Byte, 4](0b01001100, 0b01001110, 0b11000001, 0b10110000))

    writer.reset()

    # 8-byte floats
    writer.write_f64[Endian.Little](1.2345678987654321e16)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](
        0b01011000, 0b11111010, 0b01001000, 0b00110001,
        0b00101010, 0b11101110, 0b01000101, 0b01000011
    ))
    writer.reset()
    writer.write_f64[Endian.Big](1.2345678987654321e16)
    assert_equal_buffers(buf.span(length=8), InlineArray[Byte, 8](
        0b01000011, 0b01000101, 0b11101110, 0b00101010,
        0b00110001, 0b01001000, 0b11111010, 0b01011000
    ))
