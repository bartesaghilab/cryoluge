
from testing import assert_equal
from complex import ComplexFloat32

from cryoluge.image import Image, Vec
from cryoluge.fft import FFTPlan, FFTPlans, FFTImage


comptime funcs = __functions_in_module()


# NOTE: all expected FFT values computed by fftw in C++,
#       so we should get the same thing here, right?


def test_1d_f32_r2c():

    var real = Image.D1[DType.float32](sx=4)
    var fourier = FFTImage(of=real)

    real[x=0] = 1.0
    real[x=1] = 2.0
    real[x=2] = 3.0
    real[x=3] = 4.0

    var fft = FFTPlan.R2C(real=real, fourier=fourier)
    fft(real=real, fourier=fourier)

    assert_equal(fourier.complex[x=0], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier.complex[x=1], ComplexFloat32(-0.5, 0.5))
    assert_equal(fourier.complex[x=2], ComplexFloat32(-0.5, 0.0))


def test_1d_f32_c2r():

    var real = Image.D1[DType.float32](sx=4)
    var fourier = FFTImage(of=real)

    fourier.complex[x=0] = ComplexFloat32( 2.5, 0.0)
    fourier.complex[x=1] = ComplexFloat32(-0.5, 0.5)
    fourier.complex[x=2] = ComplexFloat32(-0.5, 0.0)

    var fft = FFTPlan.C2R(real=real, fourier=fourier)
    fft(real=real, fourier=fourier)

    assert_equal(real[x=0], 1.0)
    assert_equal(real[x=1], 2.0)
    assert_equal(real[x=2], 3.0)
    assert_equal(real[x=3], 4.0)


def test_2d_f32_r2c():

    var real = Image.D2[DType.float32](sx=2, sy=2)
    var fourier = FFTImage(of=real)

    real[x=0, y=0] = 1.0
    real[x=0, y=1] = 2.0
    real[x=1, y=0] = 3.0
    real[x=1, y=1] = 4.0

    var fft = FFTPlan.R2C(real=real, fourier=fourier)
    fft(real=real, fourier=fourier)

    assert_equal(fourier.complex[x=0, y=0], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier.complex[x=0, y=1], ComplexFloat32(-0.5, 0.0))
    assert_equal(fourier.complex[x=1, y=0], ComplexFloat32(-1.0, 0.0))
    assert_equal(fourier.complex[x=1, y=1], ComplexFloat32( 0.0, 0.0))


def test_2d_f32_c2r():

    var real = Image.D2[DType.float32](sx=2, sy=2)
    var fourier = FFTImage(of=real)

    fourier.complex[x=0, y=0] = ComplexFloat32( 2.5, 0.0)
    fourier.complex[x=0, y=1] = ComplexFloat32(-0.5, 0.0)
    fourier.complex[x=1, y=0] = ComplexFloat32(-1.0, 0.0)
    fourier.complex[x=1, y=1] = ComplexFloat32( 0.0, 0.0)

    var fft = FFTPlan.C2R(real=real, fourier=fourier)
    fft(real=real, fourier=fourier)

    assert_equal(real[x=0, y=0], 1.0)
    assert_equal(real[x=0, y=1], 2.0)
    assert_equal(real[x=1, y=0], 3.0)
    assert_equal(real[x=1, y=1], 4.0)


# TODO: test larger 2d transforms
# TODO: test 3d transforms


def test_1d_plans():

    # just make sure we don't trip any asserts, for now
    var plans = FFTPlans[DType.float32](Vec.D1(x=32))
    var real = plans.alloc_real()
    real.fill(0)
    var fourier = plans.alloc_fourier()
    plans.r2c(real=real, fourier=fourier)
