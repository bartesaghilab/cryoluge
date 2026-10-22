
from std.sys import Endian
from std.testing import assert_equal

from cryoluge.io import ByteBuffer, BytesWriter, Endian
from cryoluge.test import assert_equal_buffers

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_write_scalars() raises:

    var buf = ByteBuffer(8)
    var writer = BytesWriter(buf.span())

    # 1-byte integers
    writer.write_u8(5)
    assert_equal_buffers(buf.span(length=1), [5])
    writer.reset()
    writer.write_i8(-5)
    assert_equal_buffers(buf.span(length=1), [251])

    writer.reset()

    # 2-byte integers
    writer.write_u16[.little](513)
    assert_equal_buffers(buf.span(length=2), [1, 2])
    writer.reset()
    writer.write_u16[.big](513)
    assert_equal_buffers(buf.span(length=2), [2, 1])
    writer.reset()
    writer.write_i16[.little](-513)
    assert_equal_buffers(buf.span(length=2), [255, 253])
    writer.reset()
    writer.write_i16[.big](-513)
    assert_equal_buffers(buf.span(length=2), [253, 255])

    writer.reset()

    # 4-byte integers
    writer.write_u32[.little](67305985)
    assert_equal_buffers(buf.span(length=4), [1, 2, 3, 4])
    writer.reset()
    writer.write_u32[.big](67305985)
    assert_equal_buffers(buf.span(length=4), [4, 3, 2, 1])
    writer.reset()
    writer.write_i32[.little](-67305985)
    assert_equal_buffers(buf.span(length=4), [255, 253, 252, 251])
    writer.reset()
    writer.write_i32[.big](-67305985)
    assert_equal_buffers(buf.span(length=4), [251, 252, 253, 255])

    writer.reset()

    # 8-byte integers
    writer.write_u64[.little](578437695752307201)
    assert_equal_buffers(buf.span(length=8), [1, 2, 3, 4, 5, 6, 7, 8])
    writer.reset()
    writer.write_u64[.big](578437695752307201)
    assert_equal_buffers(buf.span(length=8), [8, 7, 6, 5, 4, 3, 2, 1])
    writer.reset()
    writer.write_i64[.little](-578437695752307201)
    assert_equal_buffers(buf.span(length=8), [255, 253, 252, 251, 250, 249, 248, 247])
    writer.reset()
    writer.write_i64[.big](-578437695752307201)
    assert_equal_buffers(buf.span(length=8), [247, 248, 249, 250, 251, 252, 253, 255])

    writer.reset()

    # 4-byte floats
    writer.write_f32[.little](5.42e7)
    assert_equal_buffers(buf.span(length=4), [0b10110000, 0b11000001, 0b01001110, 0b01001100])
    writer.reset()
    writer.write_f32[.big](5.42e7)
    assert_equal_buffers(buf.span(length=4), [0b01001100, 0b01001110, 0b11000001, 0b10110000])

    writer.reset()

    # 8-byte floats
    writer.write_f64[.little](1.2345678987654321e16)
    assert_equal_buffers(buf.span(length=8), [
        0b01011000, 0b11111010, 0b01001000, 0b00110001,
        0b00101010, 0b11101110, 0b01000101, 0b01000011
    ])
    writer.reset()
    writer.write_f64[.big](1.2345678987654321e16)
    assert_equal_buffers(buf.span(length=8), [
        0b01000011, 0b01000101, 0b11101110, 0b00101010,
        0b00110001, 0b01001000, 0b11111010, 0b01011000
    ])
