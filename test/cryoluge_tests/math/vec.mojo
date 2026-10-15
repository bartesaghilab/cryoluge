
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


def test_math_overloads() raises:

    comptime VI = Vec[1,.int]
    comptime VF = Vec[1,.float32]

    def update_vi(var vi: VI, func: Some[def (mut VI)]) -> VI:
        func(vi)
        return vi^

    def update_vf(var vf: VF, func: Some[def (mut VF)]) -> VF:
        func(vf)
        return vf^

    # NOTE: lambas won't compile with bodies like: `n += 2`,
    #       so we have to call eg `n.__iadd__(2)` instead

    # NOTE: there's no good way to support, eg, Float + Int
    #       either you cause ambiguous overloads, or you run into constraint hell =(

    # +
    assert_equal(VI([5]) + 2, VI([7]))
    assert_equal(VI([5]) + Int(2), VI([7]))
    assert_equal(VF([5]) + 2, VF([7]))
    assert_equal(VF([5]) + 2.0, VF([7]))
    assert_equal(VF([5]) + Float32(2.0), VF([7]))

    # +=
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__iadd__(2)), VI([7]))
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__iadd__(Int(2))), VI([7]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__iadd__(2)), VF([7]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__iadd__(2.0)), VF([7]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__iadd__(Float32(2))), VF([7]))

    # -
    assert_equal(VI([5]) - 2, VI([3]))
    assert_equal(VI([5]) - Int(2), VI([3]))
    assert_equal(VF([5]) - 2, VF([3]))
    assert_equal(VF([5]) - 2.0, VF([3]))
    assert_equal(VF([5]) - Float32(2.0), VF([3]))

    # -=
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__isub__(2)), VI([3]))
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__isub__(Int(2))), VI([3]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__isub__(2)), VF([3]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__isub__(2.0)), VF([3]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__isub__(Float32(2))), VF([3]))

    # *
    assert_equal(VI([5]) * 2, VI([10]))
    assert_equal(VI([5]) * Int(2), VI([10]))
    assert_equal(VF([5]) * 2, VF([10]))
    assert_equal(VF([5]) * 2.0, VF([10]))
    assert_equal(VF([5]) * Float32(2.0), VF([10]))

    # *=
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__imul__(2)), VI([10]))
    assert_equal(update_vi(VI([5]), lambda (mut vi: VI): vi.__imul__(Int(2))), VI([10]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__imul__(2)), VF([10]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__imul__(2.0)), VF([10]))
    assert_equal(update_vf(VF([5]), lambda (mut vf: VF): vf.__imul__(Float32(2))), VF([10]))

    # //
    assert_equal(VI([8]) // 2, VI([4]))
    assert_equal(VI([8]) // Int(2), VI([4]))
    assert_equal(VF([8]) // 2, VF([4]))
    assert_equal(VF([8]) // 2.0, VF([4]))
    assert_equal(VF([8]) // Float32(2.0), VF([4]))

    # //=
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__ifloordiv__(2)), VI([4]))
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__ifloordiv__(Int(2))), VI([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__ifloordiv__(2)), VF([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__ifloordiv__(2.0)), VF([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__ifloordiv__(Float32(2))), VF([4]))

    # /
    assert_equal(VI([8]) / 2, VI([4]))
    assert_equal(VI([8]) / Int(2), VI([4]))
    assert_equal(VF([8]) / 2, VF([4]))
    assert_equal(VF([8]) / 2.0, VF([4]))
    assert_equal(VF([8]) / Float32(2.0), VF([4]))

    # /=
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__itruediv__(2)), VI([4]))
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__itruediv__(Int(2))), VI([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__itruediv__(2)), VF([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__itruediv__(2.0)), VF([4]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__itruediv__(Float32(2))), VF([4]))

    # **
    assert_equal(VI([8]) ** 2, VI([64]))
    assert_equal(VI([8]) ** Int(2), VI([64]))
    assert_equal(VF([8]) ** 2, VF([64]))
    assert_equal(VF([8]) ** 2.0, VF([64]))
    assert_equal(VF([8]) ** Float32(2.0), VF([64]))

    # **=
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__ipow__(2)), VI([64]))
    assert_equal(update_vi(VI([8]), lambda (mut vi: VI): vi.__ipow__(Int(2))), VI([64]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__ipow__(2)), VF([64]))
    assert_equal(update_vf(VF([8]), lambda (mut vf: VF): vf.__ipow__(Int(2))), VF([64]))

    _math_overloads[.int]()
    _math_overloads[.float32]()


def _math_overloads[dtype: DType]() raises:

    comptime V = Vec[1,dtype]
    comptime S = Scalar[dtype]

    def update_v(var v: V, func: Some[def (mut V)]) -> V:
        func(v)
        return v^

    # NOTE: lambas won't compile with bodies like: `n += 2`,
    #       so we have to call eg `n.__iadd__(2)` instead

    # +
    assert_equal(V([5]) + S(2), V([7]))
    assert_equal(V([5]) + 2, V([7]))

    # +=
    assert_equal(update_v(V([5]), lambda (mut v: V): v.__iadd__(2)), V([7]))
    assert_equal(update_v(V([5]), lambda (mut v: V): v.__iadd__(S(2))), V([7]))
