
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.image import VecD


def test_ctor_accessors():

    with LexicalScope():
        var v = VecD.D1(x=5)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)

    with LexicalScope():
        var v = VecD.D2(x=5, y=42)
        assert_equal(v[0], 5)
        v[0] = 6
        assert_equal(v.x(), 6)
        assert_equal(v[1], 42)
        v[1] = 43
        assert_equal(v.y(), 43)

    with LexicalScope():
        var v = VecD.D3(x=5, y=42, z=7)
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

    assert_true(VecD.D1(x=5) == VecD.D1(x=5))
    assert_false(VecD.D1(x=5) == VecD.D1(x=6))

    assert_true(VecD.D2(x=5, y=42) == VecD.D2(x=5, y=42))
    assert_false(VecD.D2(x=5, y=42) == VecD.D2(x=6, y=42))
    assert_false(VecD.D2(x=5, y=42) == VecD.D2(x=5, y=44))
    
    assert_true(VecD.D3(x=5, y=42, z=7) == VecD.D3(x=5, y=42, z=7))
    assert_false(VecD.D3(x=5, y=42, z=7) == VecD.D3(x=6, y=42, z=7))
    assert_false(VecD.D3(x=5, y=42, z=7) == VecD.D3(x=5, y=43, z=7))
    assert_false(VecD.D3(x=5, y=42, z=7) == VecD.D3(x=5, y=42, z=8))


def test_str():
    assert_equal(String(VecD.D1(x=5)), "(5)")
    assert_equal(String(VecD.D2(x=5, y=42)), "(5, 42)")
    assert_equal(String(VecD.D3(x=5, y=42, z=7)), "(5, 42, 7)")
