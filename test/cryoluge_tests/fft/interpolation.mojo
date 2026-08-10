
from complex import ComplexScalar
from testing import assert_equal
from builtin._location import __call_location

from cryoluge.math import Vec, Matrix, EulerAnglesZYZ, complex
from cryoluge.math.units import Deg
from cryoluge.math.error import err_abs
from cryoluge.fft import FFTCoords, FFTImage, PrecomputedFFTInterpolation, PrecomputedFFTInterpolationFull, OutOfRangeBehavior, VolumeNeighborhoods
from cryoluge.test import assert_equal_float


comptime funcs = __functions_in_module()


comptime dtype = DType.float32
comptime err_fn = err_abs[dtype]
comptime Cx = ComplexScalar[dtype]
comptime Coords1 = Vec[1,Float32]
comptime Coords2 = Vec[2,Float32]
comptime Coords3 = Vec[3,Float32]
comptime ScalarInt = Scalar[DType.int]
comptime ICoords1 = Vec[1,ScalarInt]
comptime ICoords2 = Vec[2,ScalarInt]


def test_lerp_1d():

    var img = FFTImage[1,dtype](Vec[1](x=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    # exact
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-1.0)), Cx(3, -4))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=0.0)), Cx(1, 2))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=1.0)), Cx(3, 4))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=2.0)), Cx(0, 0))

    # interpolated
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-1.1)), Cx(lerp(0, 3, 0.9), lerp(0, -4, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-0.9)), Cx(lerp(3, 1, 0.1), lerp(-4, 2, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-0.5)), Cx(lerp(3, 1, 0.5), lerp(-4, 2, 0.5)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=-0.1)), Cx(lerp(3, 1, 0.9), lerp(-4, 2, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=0.1)), Cx(lerp(1, 3, 0.1), lerp(2, 4, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=0.5)), Cx(lerp(1, 3, 0.5), lerp(2, 4, 0.5)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=0.9)), Cx(lerp(1, 3, 0.9), lerp(2, 4, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords1(x=1.1)), Cx(lerp(3, 0, 0.1), lerp(4, 0, 0.1)))


def test_lerp_2d():

    var img = FFTImage[2,dtype](Vec[2](x=3, y=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    # exact

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-2.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.0, y=-2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=2.0, y=-2.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-2.0, y=-1.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.0, y=-1.0)), Cx(7, -8))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.0, y=-1.0)), Cx(9, 10))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.0, y=-1.0)), Cx(11, 12))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=2.0, y=-1.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-2.0, y=0.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.0, y=0.0)), Cx(3, -4))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.0, y=0.0)), Cx(1, 2))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.0, y=0.0)), Cx(3, 4))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=2.0, y=0.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-2.0, y=1.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.0, y=1.0)), Cx(11, -12))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.0, y=1.0)), Cx(5, 6))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.0, y=1.0)), Cx(7, 8))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=2.0, y=1.0)), Cx(0, 0))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-2.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.0, y=2.0)), Cx(0, 0))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=2.0, y=2.0)), Cx(0, 0))

    # interpolated

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=-1.1)), Cx(lerp2(0, 0, 0, 7, 0.9, 0.9), lerp2(0, 0, 0, -8, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=-1.1)), Cx(lerp2(0, 0, 7, 9, 0.1, 0.9), lerp2(0, 0, -8, 10, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=-1.1)), Cx(lerp2(0, 0, 7, 9, 0.9, 0.9), lerp2(0, 0, -8, 10, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=-1.1)), Cx(lerp2(0, 0, 9, 11, 0.1, 0.9), lerp2(0, 0, 10, 12, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=-1.1)), Cx(lerp2(0, 0, 9, 11, 0.9, 0.9), lerp2(0, 0, 10, 12, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=-1.1)), Cx(lerp2(0, 0, 11, 0, 0.1, 0.9), lerp2(0, 0, 12, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=-0.9)), Cx(lerp2(0, 7, 0, 3, 0.9, 0.1), lerp2(0, -8, 0, -4, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=-0.9)), Cx(lerp2(7, 9, 3, 1, 0.1, 0.1), lerp2(-8, 10, -4, 2, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=-0.9)), Cx(lerp2(7, 9, 3, 1, 0.9, 0.1), lerp2(-8, 10, -4, 2, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=-0.9)), Cx(lerp2(9, 11, 1, 3, 0.1, 0.1), lerp2(10, 12, 2, 4, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=-0.9)), Cx(lerp2(9, 11, 1, 3, 0.9, 0.1), lerp2(10, 12, 2, 4, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=-0.9)), Cx(lerp2(11, 0, 3, 0, 0.1, 0.1), lerp2(12, 0, 4, 0, 0.1, 0.1)))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=-0.1)), Cx(lerp2(0, 7, 0, 3, 0.9, 0.9), lerp2(0, -8, 0, -4, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=-0.1)), Cx(lerp2(7, 9, 3, 1, 0.1, 0.9), lerp2(-8, 10, -4, 2, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=-0.1)), Cx(lerp2(7, 9, 3, 1, 0.9, 0.9), lerp2(-8, 10, -4, 2, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=-0.1)), Cx(lerp2(9, 11, 1, 3, 0.1, 0.9), lerp2(10, 12, 2, 4, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=-0.1)), Cx(lerp2(9, 11, 1, 3, 0.9, 0.9), lerp2(10, 12, 2, 4, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=-0.1)), Cx(lerp2(11, 0, 3, 0, 0.1, 0.9), lerp2(12, 0, 4, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=0.1)), Cx(lerp2(0, 3, 0, 11, 0.9, 0.1), lerp2(0, -4, 0, -12, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=0.1)), Cx(lerp2(3, 1, 11, 5, 0.1, 0.1), lerp2(-4, 2, -12, 6, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=0.1)), Cx(lerp2(3, 1, 11, 5, 0.9, 0.1), lerp2(-4, 2, -12, 6, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=0.1)), Cx(lerp2(1, 3, 5, 7, 0.1, 0.1), lerp2(2, 4, 6, 8, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=0.1)), Cx(lerp2(1, 3, 5, 7, 0.9, 0.1), lerp2(2, 4, 6, 8, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=0.1)), Cx(lerp2(3, 0, 7, 0, 0.1, 0.1), lerp2(4, 0, 8, 0, 0.1, 0.1)))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=0.9)), Cx(lerp2(0, 3, 0, 11, 0.9, 0.9), lerp2(0, -4, 0, -12, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=0.9)), Cx(lerp2(3, 1, 11, 5, 0.1, 0.9), lerp2(-4, 2, -12, 6, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=0.9)), Cx(lerp2(3, 1, 11, 5, 0.9, 0.9), lerp2(-4, 2, -12, 6, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=0.9)), Cx(lerp2(1, 3, 5, 7, 0.1, 0.9), lerp2(2, 4, 6, 8, 0.1, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=0.9)), Cx(lerp2(1, 3, 5, 7, 0.9, 0.9), lerp2(2, 4, 6, 8, 0.9, 0.9)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=0.9)), Cx(lerp2(3, 0, 7, 0, 0.1, 0.9), lerp2(4, 0, 8, 0, 0.1, 0.9)))

    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-1.1, y=1.1)), Cx(lerp2(0, 11, 0, 0, 0.9, 0.1), lerp2(0, -12, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.9, y=1.1)), Cx(lerp2(11, 5, 0, 0, 0.1, 0.1), lerp2(-12, 6, 0, 0, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=-0.1, y=1.1)), Cx(lerp2(11, 5, 0, 0, 0.9, 0.1), lerp2(-12, 6, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.1, y=1.1)), Cx(lerp2(5, 7, 0, 0, 0.1, 0.1), lerp2(6, 8, 0 ,0, 0.1, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=0.9, y=1.1)), Cx(lerp2(5, 7, 0, 0, 0.9, 0.1), lerp2(6, 8, 0, 0, 0.9, 0.1)))
    assert_equal_float[err_fn](img.get(f_lerp=Coords2(x=1.1, y=1.1)), Cx(lerp2(7, 0, 0, 0, 0.1, 0.1), lerp2(8, 0, 0, 0, 0.1, 0.1)))


alias OORInterp = OutOfRangeBehavior.interpolate(ComplexScalar[dtype](5, 7))
alias OOROverride = OutOfRangeBehavior.interpolate(ComplexScalar[dtype](9, 3))


def test_plerp_i2f_1d_full():

    var img = FFTImage[1,dtype](Vec[1](x=3))
    var plerp = PrecomputedFFTInterpolationFull[1,dtype,OORInterp](img)

    assert_equal(plerp._i2f(Vec[1](x=0)), Vec[1](x=-2))
    assert_equal(plerp._i2f(Vec[1](x=1)), Vec[1](x=-1))
    assert_equal(plerp._i2f(Vec[1](x=2)), Vec[1](x=0))
    assert_equal(plerp._i2f(Vec[1](x=3)), Vec[1](x=1))


def test_plerp_f2i_1d_full():

    var img = FFTImage[1,dtype](Vec[1](x=3))
    var plerp = PrecomputedFFTInterpolationFull[1,dtype,OORInterp](img)

    assert_equal(plerp._f2i(ICoords1(x=-3)), ICoords1(x=-1))  # out of range
    assert_equal(plerp._f2i(ICoords1(x=-2)), ICoords1(x=0))
    assert_equal(plerp._f2i(ICoords1(x=-1)), ICoords1(x=1))
    assert_equal(plerp._f2i(ICoords1(x=0)), ICoords1(x=2))
    assert_equal(plerp._f2i(ICoords1(x=1)), ICoords1(x=3))
    assert_equal(plerp._f2i(ICoords1(x=2)), ICoords1(x=-1))  # out of range


def test_plerp_1d():

    var img = FFTImage[1,dtype](Vec[1](x=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[1,dtype,oor](img)

    @always_inline
    @parameter
    def check(f: Coords1):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            location=__call_location()
        )

    # exact
    check(Coords1(x=-3.0))  # out of range
    check(Coords1(x=-2.0))
    check(Coords1(x=-1.0))
    check(Coords1(x=0.0))
    check(Coords1(x=1.0))
    check(Coords1(x=2.0))  # out of range

    # interpolated
    check(Coords1(x=-2.1))  # out of range
    check(Coords1(x=-1.1))
    check(Coords1(x=-0.9))
    check(Coords1(x=-0.5))
    check(Coords1(x=-0.1))
    check(Coords1(x=0.1))
    check(Coords1(x=0.5))
    check(Coords1(x=0.9))
    check(Coords1(x=1.1))
    check(Coords1(x=2.1))  # out of range

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for x in range(NUM_SAMPLES):
        var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
        check(Coords1(x=fx))


def test_plerp_f2i_2d_full():

    var img = FFTImage[2,dtype](Vec[2](x=3, y=3))
    var plerp = PrecomputedFFTInterpolationFull[2,dtype,OORInterp](img)

    assert_equal(plerp._f2i(ICoords2(x=-2, y=-3)), ICoords2(x=0, y=-1))  # out of range
    assert_equal(plerp._f2i(ICoords2(x=-2, y=-2)), ICoords2(x=0, y=0))
    assert_equal(plerp._f2i(ICoords2(x=-2, y=-1)), ICoords2(x=0, y=1))
    assert_equal(plerp._f2i(ICoords2(x=-2, y=0)), ICoords2(x=0, y=2))
    assert_equal(plerp._f2i(ICoords2(x=-2, y=1)), ICoords2(x=0, y=3))
    assert_equal(plerp._f2i(ICoords2(x=-2, y=2)), ICoords2(x=0, y=-1))  # out of range

    assert_equal(plerp._f2i(ICoords2(x=-1, y=-3)), ICoords2(x=1, y=-1))  # out of range
    assert_equal(plerp._f2i(ICoords2(x=-1, y=-2)), ICoords2(x=1, y=0))
    assert_equal(plerp._f2i(ICoords2(x=-1, y=-1)), ICoords2(x=1, y=1))
    assert_equal(plerp._f2i(ICoords2(x=-1, y=0)), ICoords2(x=1, y=2))
    assert_equal(plerp._f2i(ICoords2(x=-1, y=1)), ICoords2(x=1, y=3))
    assert_equal(plerp._f2i(ICoords2(x=-1, y=2)), ICoords2(x=1, y=-1))  # out of range

    assert_equal(plerp._f2i(ICoords2(x=0, y=-3)), ICoords2(x=2, y=-1))  # out of range
    assert_equal(plerp._f2i(ICoords2(x=0, y=-2)), ICoords2(x=2, y=0))
    assert_equal(plerp._f2i(ICoords2(x=0, y=-1)), ICoords2(x=2, y=1))
    assert_equal(plerp._f2i(ICoords2(x=0, y=0)), ICoords2(x=2, y=2))
    assert_equal(plerp._f2i(ICoords2(x=0, y=1)), ICoords2(x=2, y=3))
    assert_equal(plerp._f2i(ICoords2(x=0, y=2)), ICoords2(x=2, y=-1))  # out of range

    assert_equal(plerp._f2i(ICoords2(x=1, y=-3)), ICoords2(x=3, y=-1))  # out of range
    assert_equal(plerp._f2i(ICoords2(x=1, y=-2)), ICoords2(x=3, y=0))
    assert_equal(plerp._f2i(ICoords2(x=1, y=-1)), ICoords2(x=3, y=1))
    assert_equal(plerp._f2i(ICoords2(x=1, y=0)), ICoords2(x=3, y=2))
    assert_equal(plerp._f2i(ICoords2(x=1, y=1)), ICoords2(x=3, y=3))
    assert_equal(plerp._f2i(ICoords2(x=1, y=2)), ICoords2(x=3, y=-1))  # out of range


def test_plerp_2d():

    var img = FFTImage[2,dtype](Vec[2](x=3, y=3))

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @always_inline
    @parameter
    def check(f: Coords2):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            location=__call_location()
        )

    # exact

    check(Coords2(x=-2.0, y=-2.0))
    check(Coords2(x=-1.0, y=-2.0))
    check(Coords2(x=0.0, y=-2.0))
    check(Coords2(x=1.0, y=-2.0))
    check(Coords2(x=2.0, y=-2.0))

    check(Coords2(x=-2.0, y=-1.0))
    check(Coords2(x=-1.0, y=-1.0))
    check(Coords2(x=0.0, y=-1.0))
    check(Coords2(x=1.0, y=-1.0))
    check(Coords2(x=2.0, y=-1.0))

    check(Coords2(x=-2.0, y=0.0))
    check(Coords2(x=-1.0, y=0.0))
    check(Coords2(x=0.0, y=0.0))
    check(Coords2(x=1.0, y=0.0))
    check(Coords2(x=2.0, y=0.0))

    check(Coords2(x=-2.0, y=1.0))
    check(Coords2(x=-1.0, y=1.0))
    check(Coords2(x=0.0, y=1.0))
    check(Coords2(x=1.0, y=1.0))
    check(Coords2(x=2.0, y=1.0))

    check(Coords2(x=-2.0, y=2.0))
    check(Coords2(x=-1.0, y=2.0))
    check(Coords2(x=0.0, y=2.0))
    check(Coords2(x=1.0, y=2.0))
    check(Coords2(x=2.0, y=2.0))

    # interpolated

    check(Coords2(x=-1.1, y=-1.1))
    check(Coords2(x=-0.9, y=-1.1))
    check(Coords2(x=-0.1, y=-1.1))
    check(Coords2(x=0.1, y=-1.1))
    check(Coords2(x=0.9, y=-1.1))
    check(Coords2(x=1.1, y=-1.1))

    check(Coords2(x=-1.1, y=-0.9))
    check(Coords2(x=-0.9, y=-0.9))
    check(Coords2(x=-0.1, y=-0.9))
    check(Coords2(x=0.1, y=-0.9))
    check(Coords2(x=0.9, y=-0.9))
    check(Coords2(x=1.1, y=-0.9))

    check(Coords2(x=-1.1, y=-0.1))
    check(Coords2(x=-0.9, y=-0.1))
    check(Coords2(x=-0.1, y=-0.1))
    check(Coords2(x=0.1, y=-0.1))
    check(Coords2(x=0.9, y=-0.1))
    check(Coords2(x=1.1, y=-0.1))

    check(Coords2(x=-1.1, y=0.1))
    check(Coords2(x=-0.9, y=0.1))
    check(Coords2(x=-0.1, y=0.1))
    check(Coords2(x=0.1, y=0.1))
    check(Coords2(x=0.9, y=0.1))
    check(Coords2(x=1.1, y=0.1))

    check(Coords2(x=-1.1, y=0.9))
    check(Coords2(x=-0.9, y=0.9))
    check(Coords2(x=-0.1, y=0.9))
    check(Coords2(x=0.1, y=0.9))
    check(Coords2(x=0.9, y=0.9))
    check(Coords2(x=1.1, y=0.9))

    check(Coords2(x=-1.1, y=1.1))
    check(Coords2(x=-0.9, y=1.1))
    check(Coords2(x=-0.1, y=1.1))
    check(Coords2(x=0.1, y=1.1))
    check(Coords2(x=0.9, y=1.1))
    check(Coords2(x=1.1, y=1.1))


def test_plerp_2d_big_odd():

    var img = FFTImage[2,dtype](Vec[2](x=7, y=7))

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        img.complex[i=i] = Cx(i*2 + 1, i*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @parameter
    def check(f: Coords2):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            msg=String("f=", f)
        )

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for y in range(NUM_SAMPLES):
        var fy = sample_range[NUM_SAMPLES, d=1](coords, y)
        for x in range(NUM_SAMPLES):
            var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
            check(Coords2(x=fx, y=fy))


def test_plerp_2d_big_even():

    var img = FFTImage[2,dtype](Vec[2](x=6, y=6))

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        img.complex[i=i] = Cx(i*2 + 1, i*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @parameter
    def check(f: Coords2):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            msg=String("f=", f)
        )

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for y in range(NUM_SAMPLES):
        var fy = sample_range[NUM_SAMPLES, d=1](coords, y)
        for x in range(NUM_SAMPLES):
            var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
            check(Coords2(x=fx, y=fy))


def test_plerp_3d_big_odd():

    var img = FFTImage[3,dtype](Vec[3](x=7, y=7, z=7))

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        img.complex[i=i] = Cx(i*2 + 1, i*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[3,dtype,oor](img)

    @parameter
    def check(f: Coords3):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            msg=String("f=", f),
            eps=1e-4
        )

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for z in range(NUM_SAMPLES):
        var fz = sample_range[NUM_SAMPLES, d=2](coords, z)
        for y in range(NUM_SAMPLES):
            var fy = sample_range[NUM_SAMPLES, d=1](coords, y)
            for x in range(NUM_SAMPLES):
                var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
                check(Coords3(x=fx, y=fy, z=fz))


def test_plerp_3d_big_even():

    var img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6))

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        img.complex[i=i] = Cx(i*2 + 1, i*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[3,dtype,oor](img)

    @parameter
    def check(f: Coords3):
        assert_equal_float[err_fn](
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            msg=String("f=", f),
            eps=1e-4
        )

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for z in range(NUM_SAMPLES):
        var fz = sample_range[NUM_SAMPLES, d=2](coords, z)
        for y in range(NUM_SAMPLES):
            var fy = sample_range[NUM_SAMPLES, d=1](coords, y)
            for x in range(NUM_SAMPLES):
                var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
                check(Coords3(x=fx, y=fy, z=fz))


def test_scan_even_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_even_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_even_more_rot_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(30, 40, 50),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_even_more_rot_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(30, 40, 50),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_even_bigger_proj_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=9, y=9)
    )


def test_scan_even_bigger_proj_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=6, y=6, z=6)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=9, y=9)
    )


def test_scan_odd_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_odd_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_odd_more_rot_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(30, 40, 50),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_odd_more_rot_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(30, 40, 50),
        sizes_real_proj = Vec[2](x=5, y=5)
    )


def test_scan_odd_more_proj_interpolate():
    _test_scan[OORInterp](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=9, y=9)
    )


def test_scan_odd_more_proj_override():
    _test_scan[OOROverride](
        img = FFTImage[3,dtype](Vec[3](x=5, y=5, z=5)),
        proj_to_volume = make_rot(5, 7, 9),
        sizes_real_proj = Vec[2](x=9, y=9)
    )


def _test_scan[
    out_of_range: OutOfRangeBehavior[dtype]
](
    *,
    var img: FFTImage[3,dtype],
    sizes_real_proj: Vec[2,Int],
    proj_to_volume: Matrix[3,3,dtype]
):

    # fill the image with arbitrary (but deterministic, and recognizable) numbers
    @parameter
    fn fill(i: Vec[3,Int]):
        var s = String(i.x(), i.y(), i.z())
        var ni = 0
        try:
            ni = atol(s)
        except:
            from os import abort
            abort(String("failed to parse int: ", s))
        var nf = Scalar[dtype](ni)
        img.complex[i=i] = Cx(re=nf, im=-nf)

    img.complex.iterate[fill]()

    # TODO: bigger simd_width
    var vol = VolumeNeighborhoods[dtype,2,out_of_range](img)

    var interp = PrecomputedFFTInterpolationFull[3,dtype,out_of_range](img)

    @parameter
    fn find(f_pi: Vec[2,Int], out results: List[Tuple[Vec[3,Scalar[dtype]],ComplexScalar[dtype]]]):

        results = List[Tuple[Vec[3,Scalar[dtype]],ComplexScalar[dtype]]]()

        @parameter
        fn filter(obs_f_pi: Vec[2,Int], out keep: Bool):
            keep = obs_f_pi == f_pi

        @parameter
        fn check(var obs_f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]):
            if obs_f_pi == f_pi:
                results.append((f_vf^, sv))

        vol.scan[filter=filter, func=check](
            proj_to_volume=proj_to_volume,
            sizes_real_proj=sizes_real_proj
        )

    @parameter
    def check(f_pi: Vec[2,Int]):

        # rotate into volume space and interpolate the volume
        var exp_f_vf = proj_to_volume*f_pi.map_scalar[dtype]().lift(z=0)
        var exp_v = interp.get(f=exp_f_vf)

        var start_dists = interp._start_dists(f=exp_f_vf)
        ref start = start_dists[0]
        var exp_i_vi = interp._f2i(f=start).map_int()

        var context = String(
            "f_pi=", f_pi,
            "  f_vi=", exp_f_vf.floor().map_int(),
            "  f_vf=", exp_f_vf,
            "  neighborhood=", interp._neighborhood(i=exp_i_vi)
        )

        # find the same value by scanning the volume
        var results = find(f_pi)

        assert_equal(
            len(results), 1,
            String("expected one sample, but got ", len(results), ". ") + context
        )

        var obs_f_vf = results[0][0].copy()
        var obs_v = results[0][1]

        assert_equal_float[err_fn](obs_f_vf, exp_f_vf, context)
        assert_equal_float[err_fn](obs_v, exp_v, context)

    # iterate the projection grid
    var coords_proj = FFTCoords(sizes_real_proj)
    for y in range(coords_proj.fmin[1](), coords_proj.fmax[1]() + 1):
        for x in range(0, coords_proj.fmax[0]() + 1):
            check(Vec[2,Int](x=x, y=y))


# NOTE: helper functions have to go after tests or the test runner won't find all the tests


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


fn sample_range[
    num_samples: Int,
    d: Int,
    dim: Int
](
    coords: FFTCoords[dim],
    i: Int
) -> Float32:

    # start with the regular frequency range
    var min = coords.fmin[d]()
    var max = coords.fmax[d]()

    # push out the bounds by one to cover the interpolatable distance
    min -= 1
    max += 1

    # and push out by one more so some samples land outside the range
    min -= 1
    max += 1

    var width = max - min
    return Float32(width*i)/Float32(num_samples - 1) + Float32(min)


fn make_rot(
    psi: Scalar[dtype],
    theta: Scalar[dtype],
    phi: Scalar[dtype],
    out rot: Matrix[3,3,dtype]
):
    rot = Matrix[3,3,dtype](uninitialized=True)
    EulerAnglesZYZ[dtype](
        psi=Deg[dtype](psi),
        theta=Deg[dtype](theta),
        phi=Deg[dtype](phi)
    ).to_matrix(mat=rot)
