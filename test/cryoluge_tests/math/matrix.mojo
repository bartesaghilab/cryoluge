
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import Matrix, Vec


comptime funcs = __functions_in_module()

comptime dtype = DType.float32
comptime M1 = Matrix.D1[dtype]
comptime M2 = Matrix.D2[dtype]
comptime M3 = Matrix.D3[dtype]
comptime V1 = Vec.D1[Scalar[dtype]]
comptime V2 = Vec.D2[Scalar[dtype]]
comptime V3 = Vec.D3[Scalar[dtype]]


def test_mul_1x1():

    var a = M1.row_major(5)
    var b = M1.row_major(7)

    assert_equal(a*b, M1.row_major(35))
    assert_equal(b*a, M1.row_major(35))


def test_mul_1x2x1():

    var a = Matrix[1,2,dtype].row_major(5, 7)
    var b = Matrix[2,1,dtype].row_major(9, 3)

    assert_equal(a*b, M1.row_major(5*9 + 3*7))

    assert_equal(b*a, M2.row_major(
        5*9, 7*9,
        5*3, 7*3
    ))


def test_transpose():

    var m1 = M1.row_major(5)
    m1.transpose()
    assert_equal(m1, M1.row_major(5))

    var m2 = M2.row_major(
        5, 6,
        7, 8
    )
    m2.transpose()
    assert_equal(m2, M2.row_major(
        5, 7,
        6, 8
    ))

    var m3 = M3.row_major(
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    )
    m3.transpose()
    assert_equal(m3, M3.row_major(
        1, 4, 7,
        2, 5, 8,
        3, 6, 9
    ))


def test_row():

    var m1 = M1.row_major(1)
    assert_equal(m1.vec(row=0), V1(x=1))

    var m2 = M2.row_major(
        1, 2,
        3, 4
    )
    assert_equal(m2.vec(row=0), V2(x=1, y=2))
    assert_equal(m2.vec(row=1), V2(x=3, y=4))

    var m3 = M3.row_major(
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    )
    assert_equal(m3.vec(row=0), V3(x=1, y=2, z=3))
    assert_equal(m3.vec(row=1), V3(x=4, y=5, z=6))
    assert_equal(m3.vec(row=2), V3(x=7, y=8, z=9))


def test_col():

    var m1 = M1.row_major(1)
    assert_equal(m1.vec(col=0), V1(x=1))

    var m2 = M2.row_major(
        1, 2,
        3, 4
    )
    assert_equal(m2.vec(col=0), V2(x=1, y=3))
    assert_equal(m2.vec(col=1), V2(x=2, y=4))

    var m3 = M3.row_major(
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    )
    assert_equal(m3.vec(col=0), V3(x=1, y=4, z=7))
    assert_equal(m3.vec(col=1), V3(x=2, y=5, z=8))
    assert_equal(m3.vec(col=2), V3(x=3, y=6, z=9))
