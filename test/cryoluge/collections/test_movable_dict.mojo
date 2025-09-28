
from testing import assert_equal, assert_raises, assert_true

from cryoluge.collections import MovableDict


@fieldwise_init
struct Thing(Movable):
    var i: Int


def test_empty():
    var dict = MovableDict[Int,Thing]()
    assert_equal(len(dict), 0)
    with assert_raises():
        ref _v = dict[5]


def test_set():
    var dict = MovableDict[Int,Thing]()
    dict[5] = Thing(42)
    assert_equal(len(dict), 1)
    assert_equal(dict[5].i, 42)


def test_pop():
    var dict = MovableDict[Int,Thing]()
    dict[5] = Thing(42)
    assert_true(dict.pop(7) is None)
    assert_equal(dict.pop(5).value().i, 42)


def test_contains():
    var dict = MovableDict[Int,Thing]()
    dict[5] = Thing(42)
    assert_true(5 in dict)
    assert_true(42 not in dict)


struct Factory:
    var num: Int
    var count: Int

    fn __init__(out self, num: Int):
        self.num = num
        self.count = 0

    fn __call__(mut self) -> Thing:
        self.count += 1
        return Thing(self.num)


def test_get_or_insert():
    var dict = MovableDict[Int,Thing]()
    var count = 0
    @parameter
    fn factory() -> Thing:
        count += 1
        return Thing(42)
    assert_equal(count, 0)
    assert_equal(dict.get_or_insert[factory](5).i, 42)
    assert_equal(count, 1)
    assert_equal(dict.get_or_insert[factory](5).i, 42)
    assert_equal(count, 1)


def test_keys():
    var dict = MovableDict[Int,Thing]()
    dict[5] = Thing(42)
    dict[7] = Thing(9)
    var iter = dict.keys()
    assert_true(iter.__has_next__())
    assert_equal(iter.__next__(), 5)
    assert_true(iter.__has_next__())
    assert_equal(iter.__next__(), 7)
    assert_true(not iter.__has_next__())


def test_key_list():
    var dict = MovableDict[Int,Thing]()
    dict[5] = Thing(42)
    dict[7] = Thing(9)
    var keys = dict.key_list()
    assert_equal(keys, [5, 7])
