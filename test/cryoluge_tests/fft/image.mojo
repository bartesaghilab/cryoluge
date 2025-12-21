
from complex import ComplexScalar
from testing import assert_equal

from cryoluge.math import Vec
from cryoluge.math.error import err_abs
from cryoluge.fft import FFTImage

from cryoluge_testlib import assert_equal_float


comptime funcs = __functions_in_module()

comptime dtype = DType.float32
comptime err_fn = err_abs[dtype]
comptime Cx = ComplexScalar[dtype]


def test_frequency_lookups_1d():

    var img = FFTImage.D1[dtype](Vec.D1(x=3))

    assert_equal(img.complex.sizes(), Vec.D1(x=2))

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    assert_equal(img.coords().fmin(), Vec.D1(x=-1))
    assert_equal(img.coords().fmax(), Vec.D1(x=1))

    assert_equal(img.get(f=Vec.D1(x=-2)),  Cx(0, 0))
    assert_equal(img.get(f=Vec.D1(x=-1)),  Cx(3, -4))
    assert_equal(img.get(f=Vec.D1(x=0)),  Cx(1, 2))
    assert_equal(img.get(f=Vec.D1(x=1)),  Cx(3, 4))
    assert_equal(img.get(f=Vec.D1(x=2)),  Cx(0, 0))


def test_frequency_lookups_2d():
        
    var img = FFTImage.D2[dtype](Vec.D2(x=3, y=3))

    assert_equal(img.complex.sizes(), Vec.D2(x=2, y=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    assert_equal(img.coords().fmin(), Vec.D2(x=-1, y=-1))
    assert_equal(img.coords().fmax(), Vec.D2(x=1, y=1))

    assert_equal(img.get(f=Vec.D2(x=-2, y=-2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=-1, y=-2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=0, y=-2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=1, y=-2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=2, y=-2)), Cx(0, 0))

    assert_equal(img.get(f=Vec.D2(x=-2, y=-1)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=-1, y=-1)), Cx(7, -8))
    assert_equal(img.get(f=Vec.D2(x=0, y=-1)), Cx(9, 10))
    assert_equal(img.get(f=Vec.D2(x=1, y=-1)), Cx(11, 12))
    assert_equal(img.get(f=Vec.D2(x=2, y=-1)), Cx(0, 0))

    assert_equal(img.get(f=Vec.D2(x=-2, y=0)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=-1, y=0)), Cx(3, -4))
    assert_equal(img.get(f=Vec.D2(x=0, y=0)), Cx(1, 2))
    assert_equal(img.get(f=Vec.D2(x=1, y=0)), Cx(3, 4))
    assert_equal(img.get(f=Vec.D2(x=2, y=0)), Cx(0, 0))

    assert_equal(img.get(f=Vec.D2(x=-2, y=1)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=-1, y=1)), Cx(11, -12))
    assert_equal(img.get(f=Vec.D2(x=0, y=1)), Cx(5, 6))
    assert_equal(img.get(f=Vec.D2(x=1, y=1)), Cx(7, 8))
    assert_equal(img.get(f=Vec.D2(x=2, y=1)), Cx(0, 0))

    assert_equal(img.get(f=Vec.D2(x=-2, y=2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=-1, y=2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=0, y=2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=1, y=2)), Cx(0, 0))
    assert_equal(img.get(f=Vec.D2(x=2, y=2)), Cx(0, 0))
