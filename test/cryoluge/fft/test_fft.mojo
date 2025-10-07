
from testing import assert_equal
from complex import ComplexFloat32

from cryoluge.image import Image, ComplexImage
from cryoluge.fft import FFTPlan


# NOTE: all expected FFT values computed by fftw in C++,
#       so we should get the same thing here, right?


def test_1d_f32_r2c():

    var real = Image.D1[DType.float32](sx=4)
    var fourier = ComplexImage.D1[DType.float32](sx=3)

    real[x=0] = 1.0
    real[x=1] = 2.0
    real[x=2] = 3.0
    real[x=3] = 4.0

    var fft = FFTPlan.R2C(real, fourier)
    fft(real, fourier)

    assert_equal(fourier[x=0], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier[x=1], ComplexFloat32(-0.5, 0.5))
    assert_equal(fourier[x=2], ComplexFloat32(-0.5, 0.0))


def test_1d_f32_c2r():

    var real = Image.D1[DType.float32](sx=4)
    var fourier = ComplexImage.D1[DType.float32](sx=3)

    fourier[x=0] = ComplexFloat32( 2.5, 0.0)
    fourier[x=1] = ComplexFloat32(-0.5, 0.5)
    fourier[x=2] = ComplexFloat32(-0.5, 0.0)

    var fft = FFTPlan.C2R(real, fourier)
    fft(real, fourier)

    assert_equal(real[x=0], 1.0)
    assert_equal(real[x=1], 2.0)
    assert_equal(real[x=2], 3.0)
    assert_equal(real[x=3], 4.0)


def test_2d_f32_r2c():

    var real = Image.D2[DType.float32](sx=2, sy=2)
    var fourier = ComplexImage.D2[DType.float32](sx=2, sy=2)

    real[x=0, y=0] = 1.0
    real[x=0, y=1] = 2.0
    real[x=1, y=0] = 3.0
    real[x=1, y=1] = 4.0

    var fft = FFTPlan.R2C(real, fourier)
    fft(real, fourier)

    assert_equal(fourier[x=0, y=0], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier[x=0, y=1], ComplexFloat32(-0.5, 0.0))
    assert_equal(fourier[x=1, y=0], ComplexFloat32(-1.0, 0.0))
    assert_equal(fourier[x=1, y=1], ComplexFloat32( 0.0, 0.0))


def test_2d_f32_c2r():

    var real = Image.D2[DType.float32](sx=2, sy=2)
    var fourier = ComplexImage.D2[DType.float32](sx=2, sy=2)

    fourier[x=0, y=0] = ComplexFloat32( 2.5, 0.0)
    fourier[x=0, y=1] = ComplexFloat32(-0.5, 0.0)
    fourier[x=1, y=0] = ComplexFloat32(-1.0, 0.0)
    fourier[x=1, y=1] = ComplexFloat32( 0.0, 0.0)

    var fft = FFTPlan.C2R(real, fourier)
    fft(real, fourier)

    assert_equal(real[x=0, y=0], 1.0)
    assert_equal(real[x=0, y=1], 2.0)
    assert_equal(real[x=1, y=0], 3.0)
    assert_equal(real[x=1, y=1], 4.0)


# TODO: test larger 2d transforms
# TODO: test 3d transforms
