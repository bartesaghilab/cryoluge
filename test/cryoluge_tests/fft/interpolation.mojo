
from complex import ComplexScalar, ComplexSIMD
from testing import assert_equal, assert_true, assert_false
from builtin._location import __call_location

from cryoluge.math import Vec, Matrix, EulerAnglesZYZ, complex
from cryoluge.math.units import Deg
from cryoluge.math.error import err_abs
from cryoluge.image.analysis import FrequencyLimits
from cryoluge.fft import FFTCoords, FFTImage, PrecomputedFFTInterpolation, PrecomputedFFTInterpolationFull, OutOfRangeBehavior, VolumeNeighborhoods, VolumeNeighborhoodsProjection
from cryoluge.fft.interpolation import _render_neighborhood, _Projections, _num_neighborhoods_in_segment, _PBound, ScanDebugger, _PIntersections
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


alias OORInterp = OutOfRangeBehavior.interpolate(ComplexScalar[dtype](999, -987))
alias OOROverride = OutOfRangeBehavior.override(ComplexScalar[dtype](999, -987))


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
            eps=1e-3,
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
            msg=String("f=", f),
            eps=1e-3
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
            msg=String("f=", f),
            eps=1e-3
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
            eps=1e-3
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
            eps=1e-3
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


# TEMP
def test_scan():

    var errors = List[String]()

    # do the big matrix of tests
    for sizes_real_vol in TestConditions.sizes_real_vols():
        for sizes_real_proj in TestConditions.sizes_real_projs():
            for rot in TestConditions.rots():
                for num_projections in TestConditions.num_projectionss():
                    @parameter
                    for simd_width in TestConditions.simd_widths():
                        @parameter
                        for oor in TestConditions.out_of_range_behaviors():
                            ref freq_limits = FrequencyLimits[dtype].none()
                            try:
                                _test_scan[simd_width, oor](
                                    sizes_real_vol,
                                    sizes_real_proj,
                                    rot,
                                    num_projections,
                                    freq_limits
                                )
                            except e:
                                errors.append(String(e))

    # test other combinations using frequency limits
    for freq_limits in TestConditions.freq_limitss():
        for num_projections in TestConditions.num_projectionss():
            @parameter
            for simd_width in TestConditions.simd_widths():
                comptime oor = TestConditions.out_of_range_behaviors()[0]
                try:
                    _test_scan[simd_width, oor](
                        TestConditions.sizes_real_vols()[0],
                        TestConditions.sizes_real_projs()[0],
                        TestConditions.rots()[0],
                        num_projections,
                        freq_limits
                    )
                except e:
                    errors.append(String(e))

    if len(errors) > 0:
        var msg = String("scanning tests failed:")
        for e in errors:
            msg += "\n" + e
        raise Error(msg)


# TEMP
# def test_segment_neighborhood():

#     var errors = List[String]()

#     for sizes_real_vol in TestConditions.sizes_real_vols():
#         @parameter
#         for simd_width in TestConditions.simd_widths():
#             @parameter
#             for oor in TestConditions.out_of_range_behaviors():
#                 try:
#                     _test_segment_neighborhood[simd_width, oor](sizes_real_vol)
#                 except e:
#                     errors.append(String(e))

#     if len(errors) > 0:
#         var msg = String("segment neighborhood tests failed:")
#         for e in errors:
#             msg += "\n" + e
#         raise Error(msg)


# TEMP
# def test_p_bounds():

#     var errors = List[String]()

#     # don't need all test conditions for this, just a few
#     for sizes_real_proj in TestConditions.sizes_real_projs():
#         for rot in TestConditions.rots():
#             @parameter
#             for simd_width in TestConditions.simd_widths():
#                 @parameter
#                 for num_projections in TestConditions.num_projectionss():
#                     try:
#                         _test_p_bounds[simd_width](
#                             sizes_real_proj,
#                             rot,
#                             num_projections
#                         )
#                     except e:
#                         errors.append(String(e))

#     if len(errors) > 0:
#         var msg = String("p bounds tests failed:")
#         for e in errors:
#             msg += "\n" + e
#         raise Error(msg)


# TODO: write dedicated test for the new grid point iterator?
# def test_new_stuff():

#     # grid_p=[ (0, -25) , (25, 24) ]
#     # rot= EulerAnglesZYZ[psi=74.60222°, theta=16.983305°, phi=66.05744°]

#     comptime rounding = 5
#     comptime out_of_range = OORInterp
#     comptime simd_width = 16

#     var sizes_real_vol = Vec[3](fill=64)
#     var sizes_real_proj = Vec[2](fill=50)
#     var rot = Vec[3](x=75, y=17, z=66)

#     var img = make_fft_image(sizes_real_vol)
#     var vol = VolumeNeighborhoods[dtype,simd_width,out_of_range](img)
#     var projections = _Projections[1,simd_width,rounding=rounding]([
#         VolumeNeighborhoodsProjection(0, _make_rot(rot))
#     ])
#     ref group = projections.groups[0]
#     var w = 0

#     @parameter
#     fn func(y_vi: Int, f_pi: Vec[2,Int]):
#         pass

#     vol.new_stuff[func,rounding=rounding](sizes_real_proj, group, w)


# NOTE: helper functions have to go after tests or the test runner won't find all the tests
#       also, having too many top-level functions apparently causes a compiler (interpreter?) crash


struct TestConditions:

    @staticmethod
    fn out_of_range_behaviors() -> List[OutOfRangeBehavior[dtype]]:
        return [
            # TEMP
            OORInterp,
            # OOROverride
        ]

    @staticmethod
    fn simd_widths() -> List[Int]:
        return [
            # TEMP
            2,
            # 4,
            # 8,
            # 16
        ]

    @staticmethod
    fn num_projectionss() -> List[Int]:
        return [
            # TEMP
            1,
            2,
            # TODO: NEXTTIME: fix bugs with multiple projections
            # 16,  # max simd_width
            # 22  # a little bit more
        ]

    @staticmethod
    fn sizes_real_vols() -> List[Vec[3,Int]]:
        return [
            # TEMP
            Vec[3](fill=6),  # even
            # Vec[3](fill=7)  # odd
        ]

    @staticmethod
    fn sizes_real_projs() -> List[Vec[2,Int]]:
        return [
            # TEMP
            Vec[2](fill=5),  # smaller
            # Vec[2](fill=9)  # bigger than volume grid, will test more out-of-range behvaior
        ]

    @staticmethod
    fn rots() -> List[Vec[3,Int]]:
        return [
            # TEMP
            # Vec[3](fill=0),  # no rotation, only +x halfspace
            # Vec[3](x=5, y=7, z=9),  # small rotation
            Vec[3](x=30, y=40, z=50),  # large rotation
            # Vec[3](x=180, y=0, z=0),  # only -x halfspace, z planes parallel
            # Vec[3](x=10, y=180 - 10, z=0)  # some -x halfspace, small rotation
            # TODO: check all 90 deg rotations!
        ]

    @staticmethod
    fn freq_limitss() -> List[FrequencyLimits[dtype]]:
        return [
            # TEMP
            # FrequencyLimits(
            #     freq_norm2_lo=Scalar[dtype](0.1),
            #     freq_norm2_hi=Scalar[dtype](0.2)
            # )
        ]


fn _make_rot(params: Vec[3,Int], out rot: Matrix[3,3,dtype]):

    # HACKHACK: many right angle rotations can't be exactly represented in the matrix,
    #           so round a bit, as needed
    comptime rounding = 5

    var rot_psi = Matrix[3,3,dtype](rotate_z=Deg[dtype](params.x()))
    if params.x() % 90 == 0:
        rot_psi = rot_psi.__round__(rounding)

    var rot_theta = Matrix[3,3,dtype](rotate_y=Deg[dtype](params.y()))
    if params.y() % 90 == 0:
        rot_theta = rot_theta.__round__(rounding)

    var rot_phi = Matrix[3,3,dtype](rotate_z=Deg[dtype](params.z()))
    if params.z() % 90 == 0:
        rot_phi = rot_phi.__round__(rounding)

    rot = rot_phi*rot_theta*rot_psi


def _test_scan[
    simd_width: Int,
    out_of_range: OutOfRangeBehavior[dtype],
](
    sizes_real_vol: Vec[3,Int],
    sizes_real_proj: Vec[2,Int],
    rot: Vec[3,Int],
    num_projections: Int,
    freq_limits: FrequencyLimits[dtype]
):

    # need to round points a bit to avoid edge cases that only matter during testing
    # in real-world use, a sample point on a boundary being included in another voxel
    # will still interpolate to nearly the same value
    comptime rounding = 5

    var img = make_fft_image(sizes_real_vol)
    var coords_proj = FFTCoords(sizes_real_proj)

    # build the volume neighborhoods (the thing we're testing!)
    var vol = VolumeNeighborhoods[dtype,simd_width,out_of_range](img)

    # build the projections
    var projections = List[VolumeNeighborhoodsProjection[dtype]](capacity=num_projections)
    fn rot_delta(p: Int) -> Vec[3,Int]:
        return Vec[3](x=5, y=6, z=7)*p
    for p in range(num_projections):
        projections.append(VolumeNeighborhoodsProjection(p, _make_rot(rot + rot_delta(p))))

    # make the older precomputed interpolation, for comparison
    var interp = PrecomputedFFTInterpolationFull[3,dtype,out_of_range](img)

    var freq_limits_checker = freq_limits.checker(sizes_real_proj)

    comptime indent = "            "
    var test_context = String(
        "\n", indent, "sizes_real_vol=", img.sizes_real,
        "\n", indent, "sizes_real_proj=", coords_proj.sizes_real(),
        "\n", indent, "rot=", rot,
        "\n", indent, "out_of_range=", out_of_range,
        "\n", indent, "freq_limits=", freq_limits.freq_norm2_lo, ",", freq_limits.freq_norm2_hi,
        "\n", indent, "simd_width=", simd_width,
        "\n", indent, "num_projections=", num_projections
    )

    # do the scan: collect all the results
    var results = List[List[_ScanResult]](length=len(projections), fill=[])
    @parameter
    fn check(proj_id: Int, var f_pi: Vec[2,Int], var f_vf: Vec[3,Scalar[dtype]], var sv: ComplexScalar[dtype]):
        results[proj_id].append(_ScanResult(proj_id, f_pi^, f_vf^, sv))
    vol.scan[check, rounding=rounding](coords_proj.sizes_real(), projections, freq_limits)

    @parameter
    fn find_results(proj_i: Int, f_pi: Vec[2,Int], out found: List[_ScanResult]):
        found = List[_ScanResult]()
        for i in range(len(results[proj_i])):
            ref result = results[proj_i][i]
            if result.f_pi == f_pi:
                found.append(result.copy())

    @parameter
    def check(f_pi: Vec[2,Int]):
        var f_pf = f_pi.map_scalar[dtype]()

        # for each projection ...
        for proj_i in range(len(projections)):
            ref proj = projections[proj_i]
            var group_i = proj_i // simd_width
            var group_offset = proj_i % simd_width

            # rotate into volume space and interpolate the volume
            var exp_f_vf = proj.proj_to_vol(f_pf)
                .round[rounding]()
            var exp_v = interp.get(f=exp_f_vf)

            # get intermediate interpolation values too
            var start_dists = interp._start_dists(f=exp_f_vf)
            var exp_f_vi = start_dists[0].map_int()
            ref dists = start_dists[1]
            var exp_neighborhood = rebind[ComplexSIMD[dtype,8]](
                interp._neighborhood(i=interp._f2i(f=exp_f_vi.map_dint()).map_int())
            )

            # get the segment coords
            var exp_f_vi_seg = exp_f_vi.copy()
            if exp_f_vi.x() < 0:
                exp_f_vi_seg = -exp_f_vi_seg - 1
            exp_f_vi_seg.x() //= vol.num_neighborhoods_in_segment
            exp_f_vi_seg.x() *= vol.num_neighborhoods_in_segment

            @parameter
            fn check_context() -> String:

                # run the scan again with a debugger
                var _debugger = ScanDebugger(proj_i, f_pi, exp_f_vi)
                @parameter
                fn debugger() -> UnsafePointer[ScanDebugger,MutAnyOrigin]:
                    return UnsafePointer(to=_debugger)
                vol.scan[check, rounding=rounding, debug=True, debugger=debugger](coords_proj.sizes_real(), projections, freq_limits)

                # render the debug log
                var debug_log = "\n" + indent + "Debug Log:"
                    + "\n" + indent + ("\n" + indent).join(_debugger.msgs)

                return test_context + String(
                    "\n", indent, "proj_i=", proj_i, " (", group_i, ",", group_offset, ")",
                    "\n", indent, "rot+delta=", rot + rot_delta(proj_i),
                    "\n", indent, "f_pi=", f_pi,
                    "\n", indent, "f_vi=", exp_f_vi,
                    "\n", indent, "f_vi_seg=", exp_f_vi_seg,
                    "\n", indent, "f_vf=", exp_f_vf,
                    "\n", indent, "dists=", dists,
                    "\n", indent, "neighborhood=", _render_neighborhood(exp_neighborhood)
                ) + debug_log

            # get the results for this projection
            var proj_results = find_results(proj_i, f_pi)

            if not freq_limits_checker.contains(f=f_pf):

                # out-of-freq-range: should get no samples
                if len(proj_results) != 0:
                    raise Error("expected zero samples, but got ", len(proj_results), ".", check_context())

            else:

                # the scan should have found this sample
                if len(proj_results) != 1:
                    raise Error("expected one sample, but got ", len(proj_results), ".", check_context())

                # check the actual interpolated value
                ref obs = proj_results[0]
                assert_equal_float[err_fn,check_context](obs.f_vf, exp_f_vf, "volume-space coordinates mismatch")
                assert_equal_float[err_fn,check_context](obs.v, exp_v, "interpolated value mismatch")

            # TEMP: extend lifetimes to work around compiler bug
            _ = proj_i
            _ = group_i
            _ = group_offset
            _ = exp_f_vi
            _ = exp_f_vi_seg
            _ = exp_f_vf
            _ = exp_neighborhood
            _ = dists
            _ = proj

    # iterate the projection grid
    for y in range(coords_proj.fmin[1](), coords_proj.fmax[1]() + 1):
        for x in range(0, coords_proj.fmax[0]() + 1):
            check(Vec[2,Int](x=x, y=y))

    # TEMP: extend lifetimes to work around compiler bug
    _ = coords_proj
    _ = vol
    _ = projections
    _ = interp
    _ = freq_limits_checker
    _ = test_context
    _ = results


@fieldwise_init
struct _ScanResult(
    Copyable,
    Movable
):
    var proj_i: Int
    var f_pi: Vec[2,Int]
    var f_vf: Vec[3,Scalar[dtype]]
    var v: ComplexScalar[dtype]


def _test_segment_neighborhood[
    simd_width: Int,
    out_of_range: OutOfRangeBehavior[dtype]
](
    sizes_real_vol: Vec[3,Int]
):

    # make an arbitrary (but simple,predictable) FFT image
    var img = make_fft_image(sizes_real_vol)
    var coords = img.coords()

    # build the volume neighborhoods (the thing we're testing!)
    var vol = VolumeNeighborhoods[dtype, simd_width, out_of_range](img)

    # make the older precomputed interpolation, for comparison
    var interp = PrecomputedFFTInterpolationFull[3,dtype,out_of_range](img)

    comptime indent = "            "
    var test_context = String(
        "\n", indent, "sizes_real_vol=", img.sizes_real,
        "\n", indent, "out_of_range=", out_of_range,
        "\n", indent, "simd_width=", simd_width
    )

    # loop over every voxel in the +x frequency range
    for fz in range(coords.fmin[2](), coords.fmax[2]() + 1):
        for fy in range(coords.fmin[1](), coords.fmax[1]() + 1):
            for fx in range(0, coords.fmax[0]() + 1, vol.num_neighborhoods_in_segment):

                var f_vi_pos = Vec[3](x=fx, y=fy, z=fz)

                @parameter
                for x_halfspace in [1, -1]:

                    var segment_neighborhood = vol._segment_neighborhood[x_halfspace](f_vi_pos)

                    var f_vi = f_vi_pos.copy()
                    @parameter
                    if x_halfspace == -1:
                        f_vi = -f_vi - 1

                    @parameter
                    for x_offset in range(vol.num_neighborhoods_in_segment):

                        var dx = Vec[3](x=x_halfspace*x_offset, y=0, z=0)
                        var f_vi_dx = f_vi + dx

                        var exp = rebind[ComplexSIMD[dtype,8]](
                            interp._neighborhood(i=interp._f2i(f_vi_dx.map_dint()).map_int())
                        )

                        var obs = segment_neighborhood.voxel_neighborhood[x_halfspace, out_of_range](x_offset)

                        if exp != obs:
                            var check_context = test_context + String(
                                "\n", indent, "f_vi_pos=", f_vi_pos,
                                "\n", indent, "x_halfspace=", x_halfspace,
                                "\n", indent, "f_vi=", f_vi,
                                "\n", indent, "x_offset=", x_offset,
                                "\n", indent, "f_vi_dx=", f_vi_dx,
                                "\n", indent, "exp=", _render_neighborhood(exp),
                                "\n", indent, "obs=", _render_neighborhood(obs)
                            )
                            assert_true(False, String("Neighborhoods don't match.") + check_context)


def _test_p_bounds[
    simd_width: Int
](
    sizes_real_proj: Vec[2,Int],
    rot: Vec[3,Int],
    num_projections: Int
):
    # need to round points a bit to avoid edge cases that only matter during testing
    # in real-world use, a sample point on a boundary being included in another voxel
    # will still interpolate to nearly the same value
    comptime rounding = 5

    # build groups out of the projections
    var projections = List[VolumeNeighborhoodsProjection[dtype]](capacity=num_projections)
    fn rot_delta(p: Int) -> Vec[3,Int]:
        return Vec[3](x=5, y=6, z=7)*p
    for p in range(num_projections):
        projections.append(VolumeNeighborhoodsProjection(p, _make_rot(rot + rot_delta(p))))
    var simd_projections = _Projections[32,simd_width,rounding=rounding](projections)

    # imagine a reference volume large enough to cover all the projection samples
    var coords_vol = FFTCoords(Vec[3](fill=sizes_real_proj.max()))

    comptime indent = "            "
    var test_context = String(
        "\n", indent, "sizes_real_proj=", sizes_real_proj,
        "\n", indent, "rot=", rot,
        "\n", indent, "simd_width=", simd_width,
        "\n", indent, "num_projections=", num_projections
    )

    # iterate over the projection grid points
    var coords_proj = FFTCoords(sizes_real_proj)
    for y in range(coords_proj.fmin_pos[1](), coords_proj.fmax[1]() + 1):
        for x in range(coords_proj.fmin_pos[0](), coords_proj.fmax[0]() + 1):

            var f_pi = Vec[2](x=x, y=y)
            var f_pf = f_pi.map_scalar[dtype]()

            # for each projection ...
            for group_i in range(len(simd_projections.groups)):
                ref proj_group = simd_projections.groups[group_i]
                for p in range(proj_group.num_projections):
                    var proj_i = Int(proj_group.proj_indices[p])
                    ref proj = projections[proj_i]

                    # rotate into volume space and discretize to the voxel
                    var f_vf = proj.proj_to_vol(f_pf)
                        .round[rounding]()
                    var f_vi_vox = f_vf.floor().map_int()

                    # get the x offset of the voxel into the segment
                    var i_vi_vox = coords_vol.f2i_contiguous(f_vi_vox)
                    comptime n = _num_neighborhoods_in_segment[simd_width]()
                    var i_vi_seg = i_vi_vox // Vec[3](x=n, y=1, z=1) 
                    var x_offset = i_vi_vox.x() % n

                    # get the x halfspace
                    var x_halfspace: Int
                    if f_vi_vox.x() >= 0:
                        x_halfspace = 1
                    else:
                        x_halfspace = -1

                    # get the segment coordinates
                    var f_vi_seg = f_vi_vox - Vec[3](x=x_halfspace*x_offset, y=0, z=0)
                    var f_vi_corner = f_vi_seg.copy()
                    if x_halfspace == -1:
                        f_vi_corner.x() -= n - 1

                    # compute the projection-space bound
                    var intersections = _PIntersections[dtype,simd_width]()
                    intersections.compute(f_vi_corner, proj_group)
                    # NOTE: since we're computing intersections directly in the -x halfspace here,
                    #       all subsequent bound calculations should pretend they're in the +x halfspace,
                    #       to avoid double-correcting for the -x halfspace
                    var bound_pf = intersections.bound_f_pf(proj_group)
                    var bound_pi = proj_group.bound_pi(bound_pf, coords_proj.fmin_pos(), coords_proj.fmax())

                    @parameter
                    fn render() -> String:
                        return proj_group.render_bound_geometry(p, f_vi_seg, coords_proj, x_halfspace=x_halfspace)

                    @parameter
                    fn debug_it() -> String:

                        # run the bound again with a debugger
                        var _debugger = ScanDebugger(proj_i, f_pi, i_vi_seg)
                        @parameter
                        fn debugger() -> UnsafePointer[ScanDebugger,MutAnyOrigin]:
                            return UnsafePointer(to=_debugger)

                        _ = intersections.bound_f_pf[debug=True, debugger=debugger](proj_group)

                        # render the debug log
                        return "\n" + indent + "Debug Log:"
                            + "\n" + indent + ("\n" + indent).join(_debugger.msgs)

                    @parameter
                    fn check_context() -> String:
                        return test_context + String(
                            "\n", indent, "f_pi=", f_pi,
                            "\n", indent, "proj_i=", proj_i, " (", group_i, ",", p, ")",
                            "\n", indent, "rot+delta=", rot + rot_delta(proj_i),
                            "\n", indent, "f_vf=", f_vf,
                            "\n", indent, "f_vi_vox=", f_vi_vox,
                            "\n", indent, "i_vi_vox=", i_vi_vox,
                            "\n", indent, "i_vi_seg=", i_vi_seg,
                            "\n", indent, "x_offset=", x_offset,
                            "\n", indent, "x_halfspace=", x_halfspace,
                            "\n", indent, "f_vi_seg=", f_vi_seg,
                            "\n", indent, "f_vi_corner=", f_vi_corner,
                            "\n", indent, "mask=", bound_pf.mask[p],
                            "\n", indent, "bound_pf=", bound_pf.f[slice=p],
                            "\n", indent, "bound_pi=", bound_pi.f[slice=p]
                        )

                    # the given bound should contain the point
                    if not bound_pi.mask[p]:
                        raise Error("No intersection with z=0"
                            + check_context()
                            + "\n" + debug_it()
                            + "\n" + render()
                        )
                    if f_pi.lt_any(bound_pi.f.min[slice=p].map_int()):
                        raise Error("Min doesn't capture sample" + check_context() + "\n" + debug_it() + "\n" + render())
                    if f_pi.gt_any(bound_pi.f.max[slice=p].map_int()):
                        raise Error("Max doesn't capture sample" + check_context() + "\n" + debug_it() + "\n" + render())

                    # compute the x-bounds for this y scanline too
                    var bound_x_pf = intersections.bound_fx_pf(proj_group, f_pi.y(), x_halfspace=1)[slice=p]
                    var bound_x_pi = proj_group.bound_pi(
                        bound_x_pf,
                        coords_proj.fmin_pos().select[0](),
                        coords_proj.fmax().select[0]()
                    )

                    @parameter
                    fn check_x_context() -> String:
                        return check_context() + String(
                            "\n", indent, "mask_x=", bound_x_pf.mask[p],
                            "\n", indent, "bound_x_pf=", bound_x_pf.f,
                            "\n", indent, "bound_x_pi=", bound_x_pi.f
                        )

                    # TEMP: need to add ProjectionGroup as an arugment here,
                    #       since relying on the closure capture trigger miscompilation bugs =(
                    from cryoluge.fft.interpolation import _ProjectionGroup
                    @parameter
                    fn debug_it_x(proj_group: _ProjectionGroup[dtype,simd_width,rounding=_]) -> String:

                        # run the bound again with a debugger
                        var _debugger = ScanDebugger(proj_i, f_pi, i_vi_seg)
                        @parameter
                        fn debugger() -> UnsafePointer[ScanDebugger,MutAnyOrigin]:
                            return UnsafePointer(to=_debugger)

                        _ = intersections.bound_fx_pf[debug=True, debugger=debugger](proj_group, f_pi.y(), x_halfspace=1)

                        # render the debug log
                        return "\n" + indent + "Debug Log x:"
                            + "\n" + indent + ("\n" + indent).join(_debugger.msgs)

                    # the given bound should contain the point
                    if not bound_x_pi.mask[0]:
                        raise Error("No intersection with scanline"
                            + check_x_context()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )
                    if f_pi.x() < Int(bound_x_pi.f.min.x()):
                        raise Error("Scanline x-min doesn't capture sample"
                            + check_x_context()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )
                    if f_pi.x() > Int(bound_x_pi.f.max.x()):
                        raise Error("Scanline x-max doesn't capture sample"
                            + check_x_context()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )

                    # the 1d bound shouldn't be bigger than the 2d bound
                    comptime eps = 1e-4  # NOTE: need a fairly large epsilon here to handle all the round off error
                    if bound_x_pf.f.min.x() + eps < bound_pf.f.min.x()[p]:
                        raise Error("Scanline x-min (float) outside of 2d min"
                            + check_x_context()
                            + "\n" + debug_it()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )
                    if bound_x_pf.f.max.x() - eps > bound_pf.f.max.x()[p]:
                        raise Error("Scanline x-max (float) outside of 2d max"
                            + check_x_context()
                            + "\n" + debug_it()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )
                    if bound_x_pi.f.min.x() < bound_pi.f.min.x()[p]:
                        raise Error("Scanline x-min (int) outside of 2d min"
                            + check_x_context()
                            + "\n" + debug_it()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )
                    if bound_x_pi.f.max.x() > bound_pi.f.max.x()[p]:
                        raise Error("Scanline x-max (int) outside of 2d max"
                            + check_x_context()
                            + "\n" + debug_it()
                            + "\n" + debug_it_x(proj_group)
                            + "\n" + render()
                        )

                    # TEMP: extend lifetimes to avoid compiler bug
                    _ = proj
                    _ = f_vf
                    _ = f_vi_vox
                    _ = i_vi_vox
                    _ = i_vi_seg
                    _ = x_offset
                    _ = x_halfspace
                    _ = f_vi_seg
                    _ = f_vi_corner
                    _ = intersections
                    _ = bound_pf
                    _ = bound_pi
                    _ = bound_x_pf
                    _ = bound_x_pi

                # TEMP: extend lifetimes to avoid compiler bug
                _ = proj_group

            # TEMP: extend lifetimes to avoid compiler bug
            _ = f_pi
            _ = f_pf

    # TEMP: extend lifetimes to avoid compiler bug
    _ = projections
    _ = simd_projections
    _ = coords_proj
    _ = test_context


fn make_fft_image(
    sizes_real: Vec[3,Int],
    out img: FFTImage[3,dtype]
):
    """Make an image with arbitrary (but deterministic, and recognizable) numbers."""

    img = FFTImage[3,dtype](sizes_real)

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
