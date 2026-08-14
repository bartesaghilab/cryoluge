
from complex import ComplexScalar
from testing import assert_equal
from builtin._location import __call_location

from cryoluge.math import Vec, Matrix, EulerAnglesZYZ, complex
from cryoluge.math.units import Deg
from cryoluge.math.error import err_abs
from cryoluge.image.analysis import FrequencyLimits
from cryoluge.fft import FFTCoords, FFTImage, PrecomputedFFTInterpolation, PrecomputedFFTInterpolationFull, OutOfRangeBehavior, VolumeNeighborhoods, VolumeNeighborhoodsProjection
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
alias OOROverride = OutOfRangeBehavior.override(ComplexScalar[dtype](9, 3))


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


def test_scan():

    var errors = List[String]()

    @parameter
    for conditions_compile in TestConditionsCompileTime.all():
        for conditions_run in TestConditionsRunTime.all():
            try:
                _test_scan[conditions_compile](conditions_run)
            except e:
                errors.append(String(e))

    if len(errors) > 0:
        var msg = String("scanning tests failed:")
        for e in errors:
            msg += "\n" + e
        raise Error(msg)


@fieldwise_init
struct TestConditionsCompileTime(
    Copyable,
    Movable
):
    var out_of_range: OutOfRangeBehavior[dtype]
    var simd_width: Int

    @staticmethod
    fn all(out list: List[Self]):
        
        var oors = [
            OORInterp,
            OOROverride
        ]

        var simd_widths = [
            2,
            4,
            8,
            16
        ]

        # iterate over the cartesian product of test parameters
        list = []
        for oor in oors:
                for simd_width in simd_widths:
                    list.append(Self(
                        oor,
                        simd_width
                    ))


@fieldwise_init
struct TestConditionsRunTime(
    Copyable,
    Movable
):
    var sizes_real_vol: Vec[3,Int]
    var sizes_real_proj: Vec[2,Int]
    var rot: Vec[3,Int]
    var freq_limits: FrequencyLimits[dtype]

    fn make_rot(self, out rot: Matrix[3,3,dtype]):
        rot = Matrix[3,3,dtype](uninitialized=True)
        EulerAnglesZYZ[dtype](
            psi=Deg[dtype](self.rot.x()),
            theta=Deg[dtype](self.rot.y()),
            phi=Deg[dtype](self.rot.z())
        ).to_matrix(mat=rot)

    @staticmethod
    fn all(out list: List[Self]):
        
        var sizes_real_vols = [
            Vec[3](fill=6),  # even
            Vec[3](fill=7)  # odd
        ]

        # the even-sized Fourier values (at z=0, z coord omitted):
        #     -3 -2 -1    0  1  2  3
        # +2  33 21 11 | 05 15 25 35  +2
        # +1  32 22 12 | 04 14 24 34  +1
        #  0  33 23 13 | 03 13 23 33   0
        #     ---------o------------
        # -1  34 24 14 | 02 12 22 32  -1
        # -2  35 25 15 | 01 11 21 31  -2
        # -3  30 20 10 | 00 10 20 30  -3
        #     -3 -2 -1    0  1  2  3

        # the odd-sized Fourier values (at z=0, z coord omitted):
        #     -3 -2 -1    0  1  2  3
        # +3  30 20 10 | 06 16 26 36  +3
        # +2  31 21 11 | 05 15 25 35  +2
        # +1  32 22 12 | 04 14 24 34  +1
        #  0  33 23 13 | 03 13 23 33   0
        #     ---------o------------
        # -1  34 24 14 | 02 12 22 32  -1
        # -2  35 25 15 | 01 11 21 31  -2
        # -3  36 26 16 | 00 10 20 30  -3
        #     -3 -2 -1    0  1  2  3

        var sizes_real_projs = [
            Vec[2](fill=5),  # smaller
            Vec[2](fill=9)  # bigger
        ]

        var rots = [
            Vec[3](fill=0),  # no rotation
            Vec[3](x=5, y=7, z=9),  # small rotation
            Vec[3](x=30, y=40, z=50)  # large rotation
            # other rotations?
        ]

        var freq_limitss = [
            FrequencyLimits[dtype].none(),
            FrequencyLimits(
                freq_norm2_lo=Scalar[dtype](0.1),
                freq_norm2_hi=Scalar[dtype](0.2)
            )
        ]

        # iterate over the cartesian product of test parameters
        list = []
        for sizes_real_vol in sizes_real_vols:
            for sizes_real_proj in sizes_real_projs:
                for rot in rots:
                    for freq_limits in freq_limitss:
                        list.append(Self(
                            sizes_real_vol.copy(),
                            sizes_real_proj.copy(),
                            rot.copy(),
                            freq_limits.copy()
                        ))


def _test_scan[conditions_compile: TestConditionsCompileTime](conditions_run: TestConditionsRunTime):

    # make an image with arbitrary (but deterministic, and recognizable) numbers
    var img = FFTImage[3,dtype](conditions_run.sizes_real_vol)
    @parameter
    fn fill(i: Vec[3,Int]):
        var f = img.coords().i2f(i=i)
        var i2 = f - img.coords().fmin_pos()
        var s = String(i2.x(), i2.y(), i2.z())
        var ni = 0
        try:
            ni = atol(s)
        except:
            from os import abort
            abort(String("failed to parse int: ", s))
        var nf = Scalar[dtype](ni)
        img.complex[i=i] = Cx(re=nf, im=-nf)

    img.complex.iterate[fill]()

    var coords_proj = FFTCoords(conditions_run.sizes_real_proj)

    # build the volume neighborhoods (the thing we're testing!)
    # with one projection
    var vol = VolumeNeighborhoods[dtype,conditions_compile.simd_width,conditions_compile.out_of_range](img)
    var rot_proj_to_vol = conditions_run.make_rot()
    var projections = [
        VolumeNeighborhoodsProjection(0, rot_proj_to_vol)
    ]

    # make the older precomputed interpolation, for comparison
    var interp = PrecomputedFFTInterpolationFull[3,dtype,conditions_compile.out_of_range](img)

    comptime indent = "            "
    var test_context = String(
        "sizes_real_vol=", img.sizes_real,
        "\n", indent, "sizes_real_proj=", coords_proj.sizes_real(),
        "\n", indent, "rot=", conditions_run.rot,
        "\n", indent, "out_of_range=", conditions_compile.out_of_range,
        "\n", indent, "freq_limits=", conditions_run.freq_limits.freq_norm2_lo, ",", conditions_run.freq_limits.freq_norm2_hi,
        "\n", indent, "simd_width=", conditions_compile.simd_width
    )

    @parameter
    fn find(f_pi: Vec[2,Int], out results: List[Tuple[Vec[3,Scalar[dtype]],ComplexScalar[dtype]]]):

        results = List[Tuple[Vec[3,Scalar[dtype]],ComplexScalar[dtype]]]()

        @parameter
        fn check(_proj_id: Int, var obs_f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]):
            if obs_f_pi == f_pi:
                results.append((f_vf^, sv))

        vol.scan[check](coords_proj.sizes_real(), projections, conditions_run.freq_limits)

    @parameter
    def check(f_pi: Vec[2,Int]):

        var check_context = test_context + String(
            "\n", indent, "f_pi=", f_pi
        )

        # find the same value by scanning the volume
        var results = find(f_pi)

        # check frequency limits
        var freq_norm2 = coords_proj.f_norm(f=f_pi.map_scalar[dtype]()).len2()
        if not conditions_run.freq_limits.contains(freq_norm2=freq_norm2):
            # should get none
            assert_equal(
                len(results), 0,
                String("expected zero samples, but got ", len(results), ". ") + check_context
            )
            return

        # rotate into volume space and interpolate the volume
        var exp_f_vf = rot_proj_to_vol*f_pi.map_scalar[dtype]().lift(z=0)
        var exp_v = interp.get(f=exp_f_vf)

        var start_dists = interp._start_dists(f=exp_f_vf)
        ref start = start_dists[0]
        var exp_i_vi = interp._f2i(f=start).map_int()

        check_context += String(
            "\n", indent, "f_vi=", exp_f_vf.floor().map_int(),
            "\n", indent, "f_vf=", exp_f_vf,
            "\n", indent, "neighborhood=", interp._neighborhood(i=exp_i_vi)
        )

        assert_equal(
            len(results), 1,
            String("expected one sample, but got ", len(results), ". ") + check_context
        )

        var obs_f_vf = results[0][0].copy()
        var obs_v = results[0][1]

        assert_equal_float[err_fn](obs_f_vf, exp_f_vf, check_context)
        assert_equal_float[err_fn](obs_v, exp_v, check_context)

    # iterate the projection grid
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
