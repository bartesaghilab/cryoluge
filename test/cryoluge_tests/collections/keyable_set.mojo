
from std.testing import assert_equal, assert_true, assert_raises

from cryoluge.collections import Keyable, KeyableSet

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


@fieldwise_init
struct Thing(Copyable, Movable, Keyable):
    var i: Int

    comptime Key = Int

    def key(self) -> Self.Key:
        return self.i


def test_init() raises:

    var set = KeyableSet[Thing](
        Thing(5),
        Thing(6)
    )

    assert_equal(len(set), 2)

    assert_equal(set[5].i, 5)
    assert_equal(set[6].i, 6)
    with assert_raises():
        _ = set[7]


def test_get() raises:

    var set = KeyableSet[Thing](
        Thing(5),
        Thing(6)
    )

    assert_equal(set.get(5).value()[].i, 5)
    assert_equal(set.get(6).value()[].i, 6)
    assert_true(set.get(7) is None)


def test_duplicate() raises:

    with assert_raises():
        var _ = KeyableSet[Thing](
            Thing(5),
            Thing(6),
            Thing(5)
        )
