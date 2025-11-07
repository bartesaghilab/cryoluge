
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import Matrix


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
