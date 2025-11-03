
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import Vec, Dimension


def test_ctor_accessors():

    with LexicalScope():
        var v = Vec.D1(x=5)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)

    with LexicalScope():
        var v = Vec.D2(x=5, y=42)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)
        assert_equal(v[1], 42)
        v[1] = 43
        assert_equal(v.y(), 43)

    with LexicalScope():
        var v = Vec.D3(x=5, y=42, z=7)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)
        assert_equal(v[1], 42)
        v[1] = 43
        assert_equal(v.y(), 43)
        assert_equal(v[2], 7)
        v[2] = 8
        assert_equal(v.z(), 8)


def test_eq():

    assert_true(Vec.D1(x=5) == Vec.D1(x=5))
    assert_false(Vec.D1(x=5) == Vec.D1(x=6))

    assert_true(Vec.D2(x=5, y=42) == Vec.D2(x=5, y=42))
    assert_false(Vec.D2(x=5, y=42) == Vec.D2(x=6, y=42))
    assert_false(Vec.D2(x=5, y=42) == Vec.D2(x=5, y=44))
    
    assert_true(Vec.D3(x=5, y=42, z=7) == Vec.D3(x=5, y=42, z=7))
    assert_false(Vec.D3(x=5, y=42, z=7) == Vec.D3(x=6, y=42, z=7))
    assert_false(Vec.D3(x=5, y=42, z=7) == Vec.D3(x=5, y=43, z=7))
    assert_false(Vec.D3(x=5, y=42, z=7) == Vec.D3(x=5, y=42, z=8))


def test_str():
    assert_equal(String(Vec.D1(x=5)), "(5)")
    assert_equal(String(Vec.D2(x=5, y=42)), "(5, 42)")
    assert_equal(String(Vec.D3(x=5, y=42, z=7)), "(5, 42, 7)")


def test_project():

    assert_equal(Vec.D1(x=5).project[Dimension.D1](), Vec.D1(x=5))
    assert_equal(Vec.D1(x=5).project_1(), Vec.D1(x=5))

    assert_equal(Vec.D2(x=5, y=42).project[Dimension.D2](), Vec.D2(x=5, y=42))
    assert_equal(Vec.D2(x=5, y=42).project_2(), Vec.D2(x=5, y=42))
    assert_equal(Vec.D2(x=5, y=42).project[Dimension.D1](), Vec.D1(x=5))
    assert_equal(Vec.D2(x=5, y=42).project_1(), Vec.D1(x=5))

    assert_equal(Vec.D3(x=5, y=42, z=7).project[Dimension.D3](), Vec.D3(x=5, y=42, z=7))
    assert_equal(Vec.D3(x=5, y=42, z=7).project[Dimension.D2](), Vec.D2(x=5, y=42))
    assert_equal(Vec.D3(x=5, y=42, z=7).project_2(), Vec.D2(x=5, y=42))
    assert_equal(Vec.D3(x=5, y=42, z=7).project[Dimension.D1](), Vec.D1(x=5))
    assert_equal(Vec.D3(x=5, y=42, z=7).project_1(), Vec.D1(x=5))
