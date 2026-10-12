
from std.testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import Vec

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_ctor_accessors() raises:

    with LexicalScope():
        var v = Vec[1](x=5)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)

    with LexicalScope():
        var v = Vec[2](x=5, y=42)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)
        assert_equal(v[1], 42)
        v[1] = 43
        assert_equal(v.y(), 43)

    with LexicalScope():
        var v = Vec[3](x=5, y=42, z=7)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)
        assert_equal(v[1], 42)
        v[1] = 43
        assert_equal(v.y(), 43)
        assert_equal(v[2], 7)
        v[2] = 8
        assert_equal(v.z(), 8)


def test_eq() raises:

    assert_true(Vec[1](x=5) == Vec[1](x=5))
    assert_false(Vec[1](x=5) == Vec[1](x=6))

    assert_true(Vec[2](x=5, y=42) == Vec[2](x=5, y=42))
    assert_false(Vec[2](x=5, y=42) == Vec[2](x=6, y=42))
    assert_false(Vec[2](x=5, y=42) == Vec[2](x=5, y=44))
    
    assert_true(Vec[3](x=5, y=42, z=7) == Vec[3](x=5, y=42, z=7))
    assert_false(Vec[3](x=5, y=42, z=7) == Vec[3](x=6, y=42, z=7))
    assert_false(Vec[3](x=5, y=42, z=7) == Vec[3](x=5, y=43, z=7))
    assert_false(Vec[3](x=5, y=42, z=7) == Vec[3](x=5, y=42, z=8))


def test_str() raises:
    assert_equal(String(Vec[1](x=5)), "(5)")
    assert_equal(String(Vec[2](x=5, y=42)), "(5, 42)")
    assert_equal(String(Vec[3](x=5, y=42, z=7)), "(5, 42, 7)")


def test_project() raises:

    assert_equal(Vec[1](x=5).project[1](), Vec[1](x=5))

    assert_equal(Vec[2](x=5, y=42).project[2](), Vec[2](x=5, y=42))
    assert_equal(Vec[2](x=5, y=42).project[1](), Vec[1](x=5))

    assert_equal(Vec[3](x=5, y=42, z=7).project[3](), Vec[3](x=5, y=42, z=7))
    assert_equal(Vec[3](x=5, y=42, z=7).project[2](), Vec[2](x=5, y=42))
    assert_equal(Vec[3](x=5, y=42, z=7).project[1](), Vec[1](x=5))


def test_lift() raises:

    assert_equal(Vec[1](x=5).lift(y=42), Vec[2](x=5, y=42))
    assert_equal(Vec[1](x=5).lift(y=42, z=7), Vec[3](x=5, y=42, z=7))

    assert_equal(Vec[2](x=5, y=42).lift(z=7), Vec[3](x=5, y=42, z=7))
