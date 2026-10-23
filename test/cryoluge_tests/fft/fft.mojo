
from std.testing import assert_equal
from std.complex import ComplexFloat32

from cryoluge.image import Image, Vec
from cryoluge.fft import FFTW, FFTImage

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


# NOTE: all expected FFT values computed by fftw in C++,
#       so we should get the same thing here, right?


def test_1d_f32_r2c() raises:

    var real = Image[1,.float32]([4], uninitialized=True)
    var fourier = FFTImage(real, uninitialized=True)

    real[[0]] = 1.0
    real[[1]] = 2.0
    real[[2]] = 3.0
    real[[3]] = 4.0

    var fftw = FFTW[.float32]()
    var plans = fftw.plans(real=real, fourier=fourier)
    plans.r2c(real=real, fourier=fourier)

    assert_equal(fourier.complex[[0]], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier.complex[[1]], ComplexFloat32(-0.5, 0.5))
    assert_equal(fourier.complex[[2]], ComplexFloat32(-0.5, 0.0))


def test_1d_f32_c2r() raises:

    var real = Image[1, .float32]([4], uninitialized=True)
    var fourier = FFTImage(real, uninitialized=True)

    fourier.complex[[0]] = ComplexFloat32( 2.5, 0.0)
    fourier.complex[[1]] = ComplexFloat32(-0.5, 0.5)
    fourier.complex[[2]] = ComplexFloat32(-0.5, 0.0)

    var fftw = FFTW[.float32]()
    var plans = fftw.plans(real=real, fourier=fourier)
    plans.c2r(real=real, fourier=fourier)

    assert_equal(real[[0]], 1.0)
    assert_equal(real[[1]], 2.0)
    assert_equal(real[[2]], 3.0)
    assert_equal(real[[3]], 4.0)


def test_2d_f32_r2c() raises:

    var real = Image[2, .float32]([2, 2], uninitialized=True)
    var fourier = FFTImage(real, uninitialized=True)

    real[[0, 0]] = 1.0
    real[[0, 1]] = 2.0
    real[[1, 0]] = 3.0
    real[[1, 1]] = 4.0

    var fftw = FFTW[.float32]()
    var plans = fftw.plans(real=real, fourier=fourier)
    plans.r2c(real=real, fourier=fourier)

    assert_equal(fourier.complex[[0, 0]], ComplexFloat32( 2.5, 0.0))
    assert_equal(fourier.complex[[0, 1]], ComplexFloat32(-0.5, 0.0))
    assert_equal(fourier.complex[[1, 0]], ComplexFloat32(-1.0, 0.0))
    assert_equal(fourier.complex[[1, 1]], ComplexFloat32( 0.0, 0.0))


def test_2d_f32_c2r() raises:

    var real = Image[2, .float32]([2, 2], uninitialized=True)
    var fourier = FFTImage(real, uninitialized=True)

    fourier.complex[[0, 0]] = ComplexFloat32( 2.5, 0.0)
    fourier.complex[[0, 1]] = ComplexFloat32(-0.5, 0.0)
    fourier.complex[[1, 0]] = ComplexFloat32(-1.0, 0.0)
    fourier.complex[[1, 1]] = ComplexFloat32( 0.0, 0.0)

    var fftw = FFTW[.float32]()
    var plans = fftw.plans(real=real, fourier=fourier)
    plans.c2r(real=real, fourier=fourier)

    assert_equal(real[[0, 0]], 1.0)
    assert_equal(real[[0, 1]], 2.0)
    assert_equal(real[[1, 0]], 3.0)
    assert_equal(real[[1, 1]], 4.0)


# TODO: test larger 2d transforms
# TODO: test 3d transforms
