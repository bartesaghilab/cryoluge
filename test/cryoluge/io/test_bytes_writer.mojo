
from testing import assert_equal
import memory

from cryoluge.io import ByteBuffer, BytesWriter


# TODO: move to test utilities?
def assert_equal_arrays(
    obs: Span[Byte],
    exp: Span[Byte]
):
    assert_equal(
        len(obs), len(exp),
        msg="Array lengths differ"
    )
    for i in range(len(exp)):
        assert_equal(
            exp[i], obs[i],
            msg=String("Arrays differ at i=", i)
        )


def test_writer_init():
    var buf = InlineArray[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    assert_equal(writer.bytes_written(), 0)
    assert_equal(writer.bytes_remaining(), 3)
    assert_equal_arrays(buf, InlineArray[Byte, 3](0, 0, 0))


def test_writer_write_once():
    var buf = InlineArray[Byte, 3](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes(InlineArray[Byte, 2](1, 2))
    assert_equal(writer.bytes_written(), 2)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_arrays(buf, InlineArray[Byte, 3](1, 2, 0))


def test_writer_write_multiple():
    var buf = InlineArray[Byte, 6](fill=0)
    var writer = BytesWriter(buf)
    writer.write_bytes(InlineArray[Byte, 2](1, 2))
    writer.write_bytes(InlineArray[Byte, 1](3))
    writer.write_bytes(InlineArray[Byte, 2](4, 5))
    assert_equal(writer.bytes_written(), 5)
    assert_equal(writer.bytes_remaining(), 1)
    assert_equal_arrays(buf, InlineArray[Byte, 6](1, 2, 3, 4, 5, 0))


def test_writer_buffer():
    var buf = ByteBuffer(3)
    var writer = BytesWriter(buf.span())
    writer.write_bytes(InlineArray[Byte, 3](1, 2, 3))
    assert_equal(buf._p[0], 1)
    assert_equal(buf._p[1], 2)
    assert_equal(buf._p[2], 3)
