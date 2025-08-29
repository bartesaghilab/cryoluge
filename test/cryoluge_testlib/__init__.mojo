
from io import FileHandle
from tempfile import NamedTemporaryFile

from testing import assert_equal


def file_handle(tempfile: NamedTemporaryFile) -> ref [tempfile._file_handle] FileHandle:
    return tempfile._file_handle
    # NOTE: _file_handle is internal, and therefore probably unstable?


def assert_equal_buffers(
    obs: Span[Byte],
    exp: Span[Byte]
):
    assert_equal(
        len(obs), len(exp),
        msg="Array lengths differ"
    )
    for i in range(len(exp)):
        assert_equal(
            obs[i], exp[i],
            msg=String("Arrays differ at i=", i)
        )
