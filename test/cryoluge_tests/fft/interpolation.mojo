
from std.complex import ComplexScalar
from std.testing import assert_equal, assert_true, assert_false

from cryoluge.math import Vec, Matrix, EulerAnglesZYZ, complex
from cryoluge.math.units import Deg
from cryoluge.math.error import err_abs
from cryoluge.image.analysis import FrequencyLimits
from cryoluge.fft import FFTCoords, FFTImage, PrecomputedFFTInterpolation, PrecomputedFFTInterpolationFull, OutOfRangeBehavior
from cryoluge.test import assert_equal_float

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


comptime dtype = DType.float32
comptime err_fn = err_abs[dtype]
comptime Cx = ComplexScalar[dtype]


def test_lerp_1d() raises:

    var img = FFTImage[1,dtype]([3], uninitialized=True)

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    # exact
    assert_equal_float(err_fn, img.get(f_lerp=[2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0]), Cx(3, -4))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0]), Cx(1, 2))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0]), Cx(3, 4))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0]), Cx(0, 0))

    # interpolated
    assert_equal_float(err_fn, img.get(f_lerp=[-1.1]), Cx(lerp(0, 3, 0.9), lerp(0, -4, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9]), Cx(lerp(3, 1, 0.1), lerp(-4, 2, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.5]), Cx(lerp(3, 1, 0.5), lerp(-4, 2, 0.5)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1]), Cx(lerp(3, 1, 0.9), lerp(-4, 2, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1]), Cx(lerp(1, 3, 0.1), lerp(2, 4, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.5]), Cx(lerp(1, 3, 0.5), lerp(2, 4, 0.5)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9]), Cx(lerp(1, 3, 0.9), lerp(2, 4, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1]), Cx(lerp(3, 0, 0.1), lerp(4, 0, 0.1)))


def test_lerp_2d() raises:

    var img = FFTImage[2,dtype]([3, 3], uninitialized=True)

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    # exact

    assert_equal_float(err_fn, img.get(f_lerp=[-2.0, -2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0, -2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0, -2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0, -2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0, -2.0]), Cx(0, 0))

    assert_equal_float(err_fn, img.get(f_lerp=[-2.0, -1.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0, -1.0]), Cx(7, -8))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0, -1.0]), Cx(9, 10))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0, -1.0]), Cx(11, 12))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0, -1.0]), Cx(0, 0))

    assert_equal_float(err_fn, img.get(f_lerp=[-2.0, 0.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0, 0.0]), Cx(3, -4))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0, 0.0]), Cx(1, 2))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0, 0.0]), Cx(3, 4))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0, 0.0]), Cx(0, 0))

    assert_equal_float(err_fn, img.get(f_lerp=[-2.0, 1.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0, 1.0]), Cx(11, -12))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0, 1.0]), Cx(5, 6))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0, 1.0]), Cx(7, 8))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0, 1.0]), Cx(0, 0))

    assert_equal_float(err_fn, img.get(f_lerp=[-2.0, 2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[-1.0, 2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[0.0, 2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[1.0, 2.0]), Cx(0, 0))
    assert_equal_float(err_fn, img.get(f_lerp=[2.0, 2.0]), Cx(0, 0))

    # interpolated

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, -1.1]), Cx(lerp2(0, 0, 0, 7, 0.9, 0.9), lerp2(0, 0, 0, -8, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, -1.1]), Cx(lerp2(0, 0, 7, 9, 0.1, 0.9), lerp2(0, 0, -8, 10, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, -1.1]), Cx(lerp2(0, 0, 7, 9, 0.9, 0.9), lerp2(0, 0, -8, 10, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, -1.1]), Cx(lerp2(0, 0, 9, 11, 0.1, 0.9), lerp2(0, 0, 10, 12, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, -1.1]), Cx(lerp2(0, 0, 9, 11, 0.9, 0.9), lerp2(0, 0, 10, 12, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, -1.1]), Cx(lerp2(0, 0, 11, 0, 0.1, 0.9), lerp2(0, 0, 12, 0, 0.1, 0.9)))

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, -0.9]), Cx(lerp2(0, 7, 0, 3, 0.9, 0.1), lerp2(0, -8, 0, -4, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, -0.9]), Cx(lerp2(7, 9, 3, 1, 0.1, 0.1), lerp2(-8, 10, -4, 2, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, -0.9]), Cx(lerp2(7, 9, 3, 1, 0.9, 0.1), lerp2(-8, 10, -4, 2, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, -0.9]), Cx(lerp2(9, 11, 1, 3, 0.1, 0.1), lerp2(10, 12, 2, 4, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, -0.9]), Cx(lerp2(9, 11, 1, 3, 0.9, 0.1), lerp2(10, 12, 2, 4, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, -0.9]), Cx(lerp2(11, 0, 3, 0, 0.1, 0.1), lerp2(12, 0, 4, 0, 0.1, 0.1)))

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, -0.1]), Cx(lerp2(0, 7, 0, 3, 0.9, 0.9), lerp2(0, -8, 0, -4, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, -0.1]), Cx(lerp2(7, 9, 3, 1, 0.1, 0.9), lerp2(-8, 10, -4, 2, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, -0.1]), Cx(lerp2(7, 9, 3, 1, 0.9, 0.9), lerp2(-8, 10, -4, 2, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, -0.1]), Cx(lerp2(9, 11, 1, 3, 0.1, 0.9), lerp2(10, 12, 2, 4, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, -0.1]), Cx(lerp2(9, 11, 1, 3, 0.9, 0.9), lerp2(10, 12, 2, 4, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, -0.1]), Cx(lerp2(11, 0, 3, 0, 0.1, 0.9), lerp2(12, 0, 4, 0, 0.1, 0.9)))

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, 0.1]), Cx(lerp2(0, 3, 0, 11, 0.9, 0.1), lerp2(0, -4, 0, -12, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, 0.1]), Cx(lerp2(3, 1, 11, 5, 0.1, 0.1), lerp2(-4, 2, -12, 6, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, 0.1]), Cx(lerp2(3, 1, 11, 5, 0.9, 0.1), lerp2(-4, 2, -12, 6, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, 0.1]), Cx(lerp2(1, 3, 5, 7, 0.1, 0.1), lerp2(2, 4, 6, 8, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, 0.1]), Cx(lerp2(1, 3, 5, 7, 0.9, 0.1), lerp2(2, 4, 6, 8, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, 0.1]), Cx(lerp2(3, 0, 7, 0, 0.1, 0.1), lerp2(4, 0, 8, 0, 0.1, 0.1)))

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, 0.9]), Cx(lerp2(0, 3, 0, 11, 0.9, 0.9), lerp2(0, -4, 0, -12, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, 0.9]), Cx(lerp2(3, 1, 11, 5, 0.1, 0.9), lerp2(-4, 2, -12, 6, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, 0.9]), Cx(lerp2(3, 1, 11, 5, 0.9, 0.9), lerp2(-4, 2, -12, 6, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, 0.9]), Cx(lerp2(1, 3, 5, 7, 0.1, 0.9), lerp2(2, 4, 6, 8, 0.1, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, 0.9]), Cx(lerp2(1, 3, 5, 7, 0.9, 0.9), lerp2(2, 4, 6, 8, 0.9, 0.9)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, 0.9]), Cx(lerp2(3, 0, 7, 0, 0.1, 0.9), lerp2(4, 0, 8, 0, 0.1, 0.9)))

    assert_equal_float(err_fn, img.get(f_lerp=[-1.1, 1.1]), Cx(lerp2(0, 11, 0, 0, 0.9, 0.1), lerp2(0, -12, 0, 0, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.9, 1.1]), Cx(lerp2(11, 5, 0, 0, 0.1, 0.1), lerp2(-12, 6, 0, 0, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[-0.1, 1.1]), Cx(lerp2(11, 5, 0, 0, 0.9, 0.1), lerp2(-12, 6, 0, 0, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.1, 1.1]), Cx(lerp2(5, 7, 0, 0, 0.1, 0.1), lerp2(6, 8, 0 ,0, 0.1, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[0.9, 1.1]), Cx(lerp2(5, 7, 0, 0, 0.9, 0.1), lerp2(6, 8, 0, 0, 0.9, 0.1)))
    assert_equal_float(err_fn, img.get(f_lerp=[1.1, 1.1]), Cx(lerp2(7, 0, 0, 0, 0.1, 0.1), lerp2(8, 0, 0, 0, 0.1, 0.1)))


comptime OORInterp = OutOfRangeBehavior.interpolate(ComplexScalar[dtype](999, -987))
comptime OOROverride = OutOfRangeBehavior.override(ComplexScalar[dtype](999, -987))


def test_plerp_i2f_1d_full() raises:

    var img = FFTImage[1,dtype]([3], uninitialized=True)
    var plerp = PrecomputedFFTInterpolationFull[1,dtype,OORInterp](img)

    assert_equal(plerp._i2f(Vec[1](x=0)), Vec[1](x=-2))
    assert_equal(plerp._i2f(Vec[1](x=1)), Vec[1](x=-1))
    assert_equal(plerp._i2f(Vec[1](x=2)), Vec[1](x=0))
    assert_equal(plerp._i2f(Vec[1](x=3)), Vec[1](x=1))


def test_plerp_f2i_1d_full() raises:

    var img = FFTImage[1,dtype]([3], uninitialized=True)
    var plerp = PrecomputedFFTInterpolationFull[1,dtype,OORInterp](img)

    assert_equal(plerp._f2i([-3]), [-1])  # out of range
    assert_equal(plerp._f2i([-2]), [0])
    assert_equal(plerp._f2i([-1]), [1])
    assert_equal(plerp._f2i([0]), [2])
    assert_equal(plerp._f2i([1]), [3])
    assert_equal(plerp._f2i([2]), [-1])  # out of range


def test_plerp_1d() raises:

    var img = FFTImage[1,dtype]([3], uninitialized=True)

    img.complex[i=0] = Cx(1, 2)  # f=(0)
    img.complex[i=1] = Cx(3, 4)  # f=(1);(-1)*

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[1,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[1,dtype]) raises {plerp, img}:
        assert_equal_float(
            err_fn, 
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f)
        )

    # exact
    check([-3.0])  # out of range
    check([-2.0])
    check([-1.0])
    check([0.0])
    check([1.0])
    check([2.0])  # out of range

    # interpolated
    check([-2.1])  # out of range
    check([-1.1])
    check([-0.9])
    check([-0.5])
    check([-0.1])
    check([0.1])
    check([0.5])
    check([0.9])
    check([1.1])
    check([2.1])  # out of range

    # sample finely in frequency space
    comptime NUM_SAMPLES = 20
    var coords = img.coords()
    for x in range(NUM_SAMPLES):
        var fx = sample_range[NUM_SAMPLES, d=0](coords, x)
        check([fx])


def test_plerp_f2i_2d_full() raises:

    var img = FFTImage[2,dtype]([3, 3], uninitialized=True)
    var plerp = PrecomputedFFTInterpolationFull[2,dtype,OORInterp](img)

    assert_equal(plerp._f2i([-2, -3]), [0, -1])  # out of range
    assert_equal(plerp._f2i([-2, -2]), [0, 0])
    assert_equal(plerp._f2i([-2, -1]), [0, 1])
    assert_equal(plerp._f2i([-2, 0]), [0, 2])
    assert_equal(plerp._f2i([-2, 1]), [0, 3])
    assert_equal(plerp._f2i([-2, 2]), [0, -1])  # out of range

    assert_equal(plerp._f2i([-1, -3]), [1, -1])  # out of range
    assert_equal(plerp._f2i([-1, -2]), [1, 0])
    assert_equal(plerp._f2i([-1, -1]), [1, 1])
    assert_equal(plerp._f2i([-1, 0]), [1, 2])
    assert_equal(plerp._f2i([-1, 1]), [1, 3])
    assert_equal(plerp._f2i([-1, 2]), [1, -1])  # out of range

    assert_equal(plerp._f2i([0, -3]), [2, -1])  # out of range
    assert_equal(plerp._f2i([0, -2]), [2, 0])
    assert_equal(plerp._f2i([0, -1]), [2, 1])
    assert_equal(plerp._f2i([0, 0]), [2, 2])
    assert_equal(plerp._f2i([0, 1]), [2, 3])
    assert_equal(plerp._f2i([0, 2]), [2, -1])  # out of range

    assert_equal(plerp._f2i([1, -3]), [3, -1])  # out of range
    assert_equal(plerp._f2i([1, -2]), [3, 0])
    assert_equal(plerp._f2i([1, -1]), [3, 1])
    assert_equal(plerp._f2i([1, 0]), [3, 2])
    assert_equal(plerp._f2i([1, 1]), [3, 3])
    assert_equal(plerp._f2i([1, 2]), [3, -1])  # out of range


def test_plerp_2d() raises:

    var img = FFTImage[2,dtype]([3, 3], uninitialized=True)

    img.complex[i=0] = Cx(1, 2)  # f=(0,0)
    img.complex[i=1] = Cx(3, 4)  # f=(1,0);(-1,0)*
    img.complex[i=2] = Cx(5, 6)  # f=(0,1)
    img.complex[i=3] = Cx(7, 8)  # f=(1,1);(-1,-1)*
    img.complex[i=4] = Cx(9, 10)  # f=(0,-1)
    img.complex[i=5] = Cx(11, 12)  # f=(1,-1);(-1,1)*

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[2,dtype]) raises {plerp, img}:
        assert_equal_float(err_fn, 
            obs=plerp.get(f=f),
            exp=img.get[or_else=oor.value](f_lerp=f),
            eps=1e-3
        )

    # exact

    check([-2.0, -2.0])
    check([-1.0, -2.0])
    check([0.0, -2.0])
    check([1.0, -2.0])
    check([2.0, -2.0])

    check([-2.0, -1.0])
    check([-1.0, -1.0])
    check([0.0, -1.0])
    check([1.0, -1.0])
    check([2.0, -1.0])

    check([-2.0, 0.0])
    check([-1.0, 0.0])
    check([0.0, 0.0])
    check([1.0, 0.0])
    check([2.0, 0.0])

    check([-2.0, 1.0])
    check([-1.0, 1.0])
    check([0.0, 1.0])
    check([1.0, 1.0])
    check([2.0, 1.0])

    check([-2.0, 2.0])
    check([-1.0, 2.0])
    check([0.0, 2.0])
    check([1.0, 2.0])
    check([2.0, 2.0])

    # interpolated

    check([-1.1, -1.1])
    check([-0.9, -1.1])
    check([-0.1, -1.1])
    check([0.1, -1.1])
    check([0.9, -1.1])
    check([1.1, -1.1])

    check([-1.1, -0.9])
    check([-0.9, -0.9])
    check([-0.1, -0.9])
    check([0.1, -0.9])
    check([0.9, -0.9])
    check([1.1, -0.9])

    check([-1.1, -0.1])
    check([-0.9, -0.1])
    check([-0.1, -0.1])
    check([0.1, -0.1])
    check([0.9, -0.1])
    check([1.1, -0.1])

    check([-1.1, 0.1])
    check([-0.9, 0.1])
    check([-0.1, 0.1])
    check([0.1, 0.1])
    check([0.9, 0.1])
    check([1.1, 0.1])

    check([-1.1, 0.9])
    check([-0.9, 0.9])
    check([-0.1, 0.9])
    check([0.1, 0.9])
    check([0.9, 0.9])
    check([1.1, 0.9])

    check([-1.1, 1.1])
    check([-0.9, 1.1])
    check([-0.1, 1.1])
    check([0.1, 1.1])
    check([0.9, 1.1])
    check([1.1, 1.1])


def test_plerp_2d_big_odd() raises:

    var img = FFTImage[2,dtype]([7, 7], uninitialized=True)

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        var f = Scalar[dtype](i)
        img.complex[i=i] = Cx(f*2 + 1, f*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[2,dtype]) raises {plerp, img}:
        assert_equal_float(
            err_fn, 
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
            check([fx, fy])


def test_plerp_2d_big_even() raises:

    var img = FFTImage[2,dtype]([6, 6], uninitialized=True)

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        var f = Scalar[dtype](i)
        img.complex[i=i] = Cx(f*2 + 1, f*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[2,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[2,dtype]) raises {plerp, img}:
        assert_equal_float(
            err_fn, 
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
            check([fx, fy])


def test_plerp_3d_big_odd() raises:

    var img = FFTImage[3,dtype]([7, 7, 7], uninitialized=True)

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        var f = Scalar[dtype](i)
        img.complex[i=i] = Cx(f*2 + 1, f*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[3,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[3,dtype]) raises {plerp, img}:
        assert_equal_float(
            err_fn, 
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
                check(([fx, fy, fz]))


def test_plerp_3d_big_even() raises:

    var img = FFTImage[3,dtype]([6, 6, 6], uninitialized=True)

    # fill the image with arbitrary (but deterministic) numbers
    for i in range(img.complex.num_pixels()):
        var f = Scalar[dtype](i)
        img.complex[i=i] = Cx(f*2 + 1, f*2 + 2)

    comptime oor = OORInterp
    var plerp = PrecomputedFFTInterpolation[3,dtype,oor](img)

    @inline(.always)
    def check(f: Vec[3,dtype]) raises  {plerp, img}:
        assert_equal_float(
            err_fn, 
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
                check([fx, fy, fz])


def lerp(v0: Scalar[dtype], v1: Scalar[dtype], t: Scalar[dtype], out v: Scalar[dtype]):
    v = v0*(1 - t) + t*v1


def lerp2(
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


def sample_range[
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
