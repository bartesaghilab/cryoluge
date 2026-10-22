
from std.testing import assert_equal

from cryoluge.math import Vec
from cryoluge.math.units import Ang, Px

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_math_scalar() raises:

    var i = Ang(5)

    # just spot-check some of the math operators

    assert_equal(i + 3, Ang(8))
    assert_equal(i + Int(3), Ang(8))
    assert_equal(i + Ang(3), Ang(8))

    assert_equal(i > 0, True)
    assert_equal(i > Int(0), True)
    assert_equal(i > Ang(0), True)


    var f = Ang(5.0)

    # just spot-check some of the math operators

    assert_equal(f + 3, Ang(8.0))
    assert_equal(f + 3.0, Ang(8.0))
    assert_equal(f + Float64(3), Ang(8.0))
    assert_equal(f + Ang(3.0), Ang(8.0))

    assert_equal(f > 0, True)
    assert_equal(f > 0.0, True)
    assert_equal(f > Float64(0), True)
    assert_equal(f > Ang(0.0), True)


def test_math_vector() raises:

    var i = Ang(Vec[2](x=5, y=6))

    assert_equal(i + 3, Ang(Vec[2](x=8, y=9)))
    assert_equal(i + Int(3), Ang(Vec[2](x=8, y=9)))
    assert_equal(i + Ang(3), Ang(Vec[2](x=8, y=9)))
    assert_equal(i + Ang(Vec[2](fill=3)), Ang(Vec[2](x=8, y=9)))

    var f = Ang(Vec[2](x=5.0, y=6.0))

    assert_equal(f + 3.0, Ang(Vec[2](x=8.0, y=9.0)))
    assert_equal(f + Float64(3), Ang(Vec[2](x=8.0, y=9.0)))
    assert_equal(f + Ang(Vec[2](fill=3.0)), Ang(Vec[2](x=8.0, y=9.0)))


def test_vec_access() raises:

    var i = Ang(Vec[2](x=5, y=9))
    var i0 = 0
    var i1 = 1

    assert_equal(i.x(), Ang(5))
    assert_equal(i.y(), Ang(9))
    assert_equal(i[0], Ang(5))
    assert_equal(i[1], Ang(9))
    assert_equal(i[i0], Ang(5))
    assert_equal(i[i1], Ang(9))

    i = Ang(Vec[2](fill=0))
    i[0] = Ang(6)
    assert_equal(i.value, [6, 0])

    i = Ang(Vec[2](fill=0))
    i[1] = Ang(6)
    assert_equal(i.value, [0, 6])

    i = Ang(Vec[2](fill=0))
    i[1] = 7
    assert_equal(i.value, [0, 7])

    i = Ang(Vec[2](fill=0))
    i[i0] = 7
    assert_equal(i.value, [7, 0])


def test_to_dtype() raises:

    # scalars
    assert_equal(
        Ang(5.0).to_dtype[.float64](),
        Ang(Float64(5.0))
    )

    # vectors
    assert_equal(
        Ang(Vec[2](x=5.0, y=7.0)).to_dtype[.float64](),
        Ang(Vec[2](x=Float64(5), y=Float64(7)))
    )


def test_ang_px() raises:

    var pixel_size = 1.5

    # scalars
    assert_equal(Ang(6.0).to_px(pixel_size), Px(4.0))
    assert_equal(Px(4.0).to_ang(pixel_size), Ang(6.0))

    # vectors
    assert_equal(Ang(Vec[2](x=6.0, y=9.0)).to_px(pixel_size), Px(Vec[2](x=4.0, y=6.0)))
    assert_equal(Px(Vec[2](x=4.0, y=6.0)).to_ang(pixel_size), Ang(Vec[2](x=6.0, y=9.0)))
