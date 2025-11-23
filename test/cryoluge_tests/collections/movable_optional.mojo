
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.collections import MovableOptional


comptime funcs = __functions_in_module()


@fieldwise_init
struct Thing(Movable):
    var i: Int


def test_empty():
    var opt = MovableOptional[Thing]()
    assert_true(opt is None)
    assert_false(opt is not None)


def test_full():
    var opt = MovableOptional[Thing](Thing(5))
    assert_false(opt is None)
    assert_true(opt is not None)


def test_value():
    var opt = MovableOptional[Thing](Thing(5))
    assert_equal(opt.value().i, 5)
    assert_true(opt is not None)


def test_take():
    var opt = MovableOptional[Thing](Thing(5))
    var v: Thing = opt.take()
    assert_true(opt is None)
    assert_equal(v.i, 5)


def test_or_else():
    @parameter
    fn defaulter() -> Thing:
        return Thing(42)
    var opt = MovableOptional[Thing](Thing(5))
    assert_equal(opt.or_else[defaulter]().i, 5)
    opt = MovableOptional[Thing]()
    assert_equal(opt.or_else[defaulter]().i, 42)


def test_map():

    @parameter
    fn mapper(v: Thing) -> Int:
        return v.i
    
    # NOTE: infer-only parameters in functions don't seem to work?

    with LexicalScope():
        var v = MovableOptional[Thing](Thing(5))
            .map[Int, mapper]()
        assert_true(v is not None)
        assert_equal(v.value(), 5)

    with LexicalScope():
        var v = MovableOptional[Thing]()
            .map[Int, mapper]()
        assert_true(v is None)


def test_implicit():

    with LexicalScope():
        var opt: MovableOptional[Thing] = None
        assert_true(opt is None)

    with LexicalScope():
        var opt: MovableOptional[Thing] = Thing(5)
        assert_true(opt is not None)
        assert_equal(opt.value().i, 5)


@fieldwise_init
struct Crasher:
    var thing: MovableOptional[Thing]

def test_crash():
    var crasher = Crasher(None)
    var _ = crasher^


def test_unwrap():
    var opt = MovableOptional[Thing](Thing(5))
    var thing = opt^.unwrap()
    assert_equal(thing.i, 5)
