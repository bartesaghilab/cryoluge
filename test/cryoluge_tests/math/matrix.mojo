
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import Matrix


comptime funcs = __functions_in_module()


def test_mul_1x1():

    var a = Matrix[1,1,DType.float32](fill=0)
    a[0] = InlineArray[Float32,1](5)
    var b = Matrix[1,1,DType.float32](fill=0)
    b[0] = InlineArray[Float32,1](7)

    var ab = a*b
    assert_equal(ab[0,0], 35)

    var ba = b*a
    assert_equal(ba[0,0], 35)


def test_mul_1x2x1():

    var a = Matrix[1,2,DType.float32](fill=0)
    a[0] = InlineArray[Float32,2](5, 7)
    var b = Matrix[2,1,DType.float32](fill=0)
    b[0] = InlineArray[Float32,1](9)
    b[1] = InlineArray[Float32,1](3)

    var ab = a*b
    assert_equal(ab[0,0], 5*9 + 3*7)

    var ba = b*a
    assert_equal(ba[0,0], 5*9)
    assert_equal(ba[0,1], 7*9)
    assert_equal(ba[1,0], 5*3)
    assert_equal(ba[1,1], 7*3)


def test_transpose_1x1():

    var a = Matrix[1,1,DType.float32](fill=0)
    a[0,0] = 5

    a.transpose()

    assert_equal(a[0,0], 5)


def test_transpose_2x2():

    var a = Matrix[2,2,DType.float32](fill=0)
    a[0,0] = 5
    a[0,1] = 6
    a[1,0] = 7
    a[1,1] = 8

    a.transpose()

    assert_equal(a[0,0], 5)
    assert_equal(a[0,1], 7)
    assert_equal(a[1,0], 6)
    assert_equal(a[1,1], 8)


def test_transpose_3x3():

    var a = Matrix[3,3,DType.float32](fill=0)
    a[0,0] = 1
    a[0,1] = 2
    a[0,2] = 3
    a[1,0] = 4
    a[1,1] = 5
    a[1,2] = 6
    a[2,0] = 7
    a[2,1] = 8
    a[2,2] = 9

    a.transpose()

    assert_equal(a[0,0], 1)
    assert_equal(a[0,1], 4)
    assert_equal(a[0,2], 7)
    assert_equal(a[1,0], 2)
    assert_equal(a[1,1], 5)
    assert_equal(a[1,2], 8)
    assert_equal(a[2,0], 3)
    assert_equal(a[2,1], 6)
    assert_equal(a[2,2], 9)
