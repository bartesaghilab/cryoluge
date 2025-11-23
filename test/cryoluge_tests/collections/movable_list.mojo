
from testing import assert_equal, assert_true

from cryoluge.collections import MovableList


comptime funcs = __functions_in_module()


@fieldwise_init
struct Thing(Movable, EqualityComparable, Stringable):
    # NOTE: *NOT* Copyable or ImplicitlyCopyable
    var i: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.i == other.i

    fn __str__(self) -> String:
        return String('Thing[', self.i, ']')


def test_empty():
    var list = MovableList[Thing]()
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 0)


def test_append():

    var list = MovableList[Thing]()

    list.append(Thing(5))
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 1)
    assert_equal(list[0], Thing(5))

    list.append(Thing(42))
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 2)
    assert_equal(list[0], Thing(5))
    assert_equal(list[1], Thing(42))


def test_append_expand():

    var list = MovableList[Thing]()

    list.append(Thing(5))
    list.append(Thing(6))
    list.append(Thing(7))
    list.append(Thing(8))

    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 4)

    # the next append should trigger an expand
    list.append(Thing(42))
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY*2)
    assert_equal(len(list), 5)
    assert_equal(list[0], Thing(5))
    assert_equal(list[1], Thing(6))
    assert_equal(list[2], Thing(7))
    assert_equal(list[3], Thing(8))
    assert_equal(list[4], Thing(42))


def test_remove_indexed():

    var list = MovableList[Thing]()
    list.append(Thing(5))
    list.append(Thing(6))
    list.append(Thing(7))
    list.append(Thing(8))

    assert_equal(list.remove(3), Thing(8))

    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 3)
    assert_equal(list[0], Thing(5))
    assert_equal(list[1], Thing(6))
    assert_equal(list[2], Thing(7))

    assert_equal(list.remove(1), Thing(6))
    
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 2)
    assert_equal(list[0], Thing(5))
    assert_equal(list[1], Thing(7))

    assert_equal(list.remove(0), Thing(5))
    
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 1)
    assert_equal(list[0], Thing(7))

    assert_equal(list.remove(0), Thing(7))
    
    assert_equal(list.capacity(), MovableList.DEFAULT_CAPACITY)
    assert_equal(len(list), 0)


def test_find():

    var list = MovableList[Thing]()
    list.append(Thing(5))
    list.append(Thing(6))
    list.append(Thing(7))
    list.append(Thing(8))

    assert_equal(list.find(list[0]).value(), 0)
    assert_equal(list.find(list[1]).value(), 1)
    assert_equal(list.find(list[2]).value(), 2)
    assert_equal(list.find(list[3]).value(), 3)
    assert_true(list.find(Thing(5)) is None)


def test_first():

    var list = MovableList[Thing]()
    list.append(Thing(5))
    list.append(Thing(6))
    list.append(Thing(7))
    list.append(Thing(8))

    fn is_n[n: Int](t: Thing) -> Bool:
        return t.i == n
    assert_equal(list.first[is_n[5]]().value(), 0)
    assert_equal(list.first[is_n[6]]().value(), 1)
    assert_equal(list.first[is_n[7]]().value(), 2)
    assert_equal(list.first[is_n[8]]().value(), 3)
    assert_true(list.first[is_n[9]]() is None)
