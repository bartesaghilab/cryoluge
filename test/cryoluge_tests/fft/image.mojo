
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


def test_lerp_1d():

    var img = FFTImage.D1[dtype](Vec.D1(x=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    # exact
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-1.0)), Cx(3, -4))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=0.0)), Cx(1, 2))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=1.0)), Cx(3, 4))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=2.0)), Cx(0, 0))

    # interpolated
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-1.1)), Cx(lerp(0, 3, 0.9), lerp(0, -4, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-0.9)), Cx(lerp(3, 1, 0.1), lerp(-4, 2, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-0.5)), Cx(lerp(3, 1, 0.5), lerp(-4, 2, 0.5)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=-0.1)), Cx(lerp(3, 1, 0.9), lerp(-4, 2, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=0.1)), Cx(lerp(1, 3, 0.1), lerp(2, 4, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=0.5)), Cx(lerp(1, 3, 0.5), lerp(2, 4, 0.5)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=0.9)), Cx(lerp(1, 3, 0.9), lerp(2, 4, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D1(x=1.1)), Cx(lerp(3, 0, 0.1), lerp(4, 0, 0.1)))


def test_lerp_2d():

    var img = FFTImage.D2[dtype](Vec.D2(x=3, y=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    # exact

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-2.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=2.0, y=-2.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-2.0, y=-1.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.0, y=-1.0)), Cx(7, -8))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.0, y=-1.0)), Cx(9, 10))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.0, y=-1.0)), Cx(11, 12))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=2.0, y=-1.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-2.0, y=0.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.0, y=0.0)), Cx(3, -4))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.0, y=0.0)), Cx(1, 2))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.0, y=0.0)), Cx(3, 4))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=2.0, y=0.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-2.0, y=1.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.0, y=1.0)), Cx(11, -12))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.0, y=1.0)), Cx(5, 6))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.0, y=1.0)), Cx(7, 8))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=2.0, y=1.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-2.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=2.0, y=2.0)), Cx(0, 0))

    # interpolated

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=-1.1)), Cx(lerp2(0, 0, 0, 7, 0.9, 0.9), lerp2(0, 0, 0, -8, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=-1.1)), Cx(lerp2(0, 0, 7, 9, 0.1, 0.9), lerp2(0, 0, -8, 10, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=-1.1)), Cx(lerp2(0, 0, 7, 9, 0.9, 0.9), lerp2(0, 0, -8, 10, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=-1.1)), Cx(lerp2(0, 0, 9, 11, 0.1, 0.9), lerp2(0, 0, 10, 12, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=-1.1)), Cx(lerp2(0, 0, 9, 11, 0.9, 0.9), lerp2(0, 0, 10, 12, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=-1.1)), Cx(lerp2(0, 0, 11, 0, 0.1, 0.9), lerp2(0, 0, 12, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=-0.9)), Cx(lerp2(0, 7, 0, 3, 0.9, 0.1), lerp2(0, -8, 0, -4, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=-0.9)), Cx(lerp2(7, 9, 3, 1, 0.1, 0.1), lerp2(-8, 10, -4, 2, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=-0.9)), Cx(lerp2(7, 9, 3, 1, 0.9, 0.1), lerp2(-8, 10, -4, 2, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=-0.9)), Cx(lerp2(9, 11, 1, 3, 0.1, 0.1), lerp2(10, 12, 2, 4, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=-0.9)), Cx(lerp2(9, 11, 1, 3, 0.9, 0.1), lerp2(10, 12, 2, 4, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=-0.9)), Cx(lerp2(11, 0, 3, 0, 0.1, 0.1), lerp2(12, 0, 4, 0, 0.1, 0.1)))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=-0.1)), Cx(lerp2(0, 7, 0, 3, 0.9, 0.9), lerp2(0, -8, 0, -4, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=-0.1)), Cx(lerp2(7, 9, 3, 1, 0.1, 0.9), lerp2(-8, 10, -4, 2, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=-0.1)), Cx(lerp2(7, 9, 3, 1, 0.9, 0.9), lerp2(-8, 10, -4, 2, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=-0.1)), Cx(lerp2(9, 11, 1, 3, 0.1, 0.9), lerp2(10, 12, 2, 4, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=-0.1)), Cx(lerp2(9, 11, 1, 3, 0.9, 0.9), lerp2(10, 12, 2, 4, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=-0.1)), Cx(lerp2(11, 0, 3, 0, 0.1, 0.9), lerp2(12, 0, 4, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=0.1)), Cx(lerp2(0, 3, 0, 11, 0.9, 0.1), lerp2(0, -4, 0, -12, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=0.1)), Cx(lerp2(3, 1, 11, 5, 0.1, 0.1), lerp2(-4, 2, -12, 6, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=0.1)), Cx(lerp2(3, 1, 11, 5, 0.9, 0.1), lerp2(-4, 2, -12, 6, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=0.1)), Cx(lerp2(1, 3, 5, 7, 0.1, 0.1), lerp2(2, 4, 6, 8, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=0.1)), Cx(lerp2(1, 3, 5, 7, 0.9, 0.1), lerp2(2, 4, 6, 8, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=0.1)), Cx(lerp2(3, 0, 7, 0, 0.1, 0.1), lerp2(4, 0, 8, 0, 0.1, 0.1)))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=0.9)), Cx(lerp2(0, 3, 0, 11, 0.9, 0.9), lerp2(0, -4, 0, -12, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=0.9)), Cx(lerp2(3, 1, 11, 5, 0.1, 0.9), lerp2(-4, 2, -12, 6, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=0.9)), Cx(lerp2(3, 1, 11, 5, 0.9, 0.9), lerp2(-4, 2, -12, 6, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=0.9)), Cx(lerp2(1, 3, 5, 7, 0.1, 0.9), lerp2(2, 4, 6, 8, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=0.9)), Cx(lerp2(1, 3, 5, 7, 0.9, 0.9), lerp2(2, 4, 6, 8, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=0.9)), Cx(lerp2(3, 0, 7, 0, 0.1, 0.9), lerp2(4, 0, 8, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-1.1, y=1.1)), Cx(lerp2(0, 11, 0, 0, 0.9, 0.1), lerp2(0, -12, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.9, y=1.1)), Cx(lerp2(11, 5, 0, 0, 0.1, 0.1), lerp2(-12, 6, 0, 0, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=-0.1, y=1.1)), Cx(lerp2(11, 5, 0, 0, 0.9, 0.1), lerp2(-12, 6, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.1, y=1.1)), Cx(lerp2(5, 7, 0, 0, 0.1, 0.1), lerp2(6, 8, 0 ,0, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=0.9, y=1.1)), Cx(lerp2(5, 7, 0, 0, 0.9, 0.1), lerp2(6, 8, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Vec.D2(x=1.1, y=1.1)), Cx(lerp2(7, 0, 0, 0, 0.1, 0.1), lerp2(8, 0, 0, 0, 0.1, 0.1)))

    
fn lerp(v0: Scalar[dtype], v1: Scalar[dtype], t: Scalar[dtype], out v: Scalar[dtype]):
    v = v0*(1 - t) + t*v1


fn lerp2(
    v00: Scalar[dtype],
    v10: Scalar[dtype],
    v01: Scalar[dtype],
    v11: Scalar[dtype],
    t0: Scalar[dtype],
    t1: Scalar[dtype],
    out v: Scalar[dtype]
):
    v = lerp(
        lerp(v00, v10, t0),
        lerp(v01, v11, t0),
        t1
    )
