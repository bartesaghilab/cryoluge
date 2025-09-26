
from testing import assert_equal, assert_true, assert_raises

from cryoluge.collections import Keyable, KeyableSet


@fieldwise_init
struct Thing(Copyable, Movable, Keyable):
    var i: Int

    alias Key = Int

    fn key(self) -> Int:
        return self.i


def test_init():

    var set = KeyableSet[Thing](
        Thing(5),
        Thing(6)
    )

    assert_equal(len(set), 2)
    assert_equal(set[5].value()[].i, 5)
    assert_equal(set[6].value()[].i, 6)
    assert_true(set[7] is None)


def test_duplicate():

    with assert_raises():
        var _ = KeyableSet[Thing](
            Thing(5),
            Thing(6),
            Thing(5)
        )
