
from testing import assert_equal


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
