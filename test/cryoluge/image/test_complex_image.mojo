
from testing import assert_equal
from complex import ComplexFloat32

from cryoluge.image import Image, ComplexImage


def test_multiply_4():
    # should be evenly-divisible by one vector instruction, with no draining

    var fourier = ComplexImage.D1[DType.float32](sx=4)

    fourier[x=0] = ComplexFloat32(1.0, 2.0)
    fourier[x=1] = ComplexFloat32(3.0, 4.0)
    fourier[x=2] = ComplexFloat32(5.0, 6.0)
    fourier[x=3] = ComplexFloat32(7.0, 8.0)

    fourier.multiply[4](3.0)

    assert_equal(fourier[x=0], ComplexFloat32( 3.0,  6.0))
    assert_equal(fourier[x=1], ComplexFloat32( 9.0, 12.0))
    assert_equal(fourier[x=2], ComplexFloat32(15.0, 18.0))
    assert_equal(fourier[x=3], ComplexFloat32(21.0, 24.0))


def test_multiply_5():
    # should require one vector instruction, and drain with one scalar instruction

    var fourier = ComplexImage.D1[DType.float32](sx=5)

    fourier[x=0] = ComplexFloat32(1.0, 2.0)
    fourier[x=1] = ComplexFloat32(3.0, 4.0)
    fourier[x=2] = ComplexFloat32(5.0, 6.0)
    fourier[x=3] = ComplexFloat32(7.0, 8.0)
    fourier[x=4] = ComplexFloat32(9.0, 10.0)

    fourier.multiply[4](3.0)

    assert_equal(fourier[x=0], ComplexFloat32( 3.0,  6.0))
    assert_equal(fourier[x=1], ComplexFloat32( 9.0, 12.0))
    assert_equal(fourier[x=2], ComplexFloat32(15.0, 18.0))
    assert_equal(fourier[x=3], ComplexFloat32(21.0, 24.0))
    assert_equal(fourier[x=4], ComplexFloat32(27.0, 30.0))


def test_multiply_max():
    # should require one vector instruction, with no draining

    alias max_width = ComplexImage.D1[DType.float32].pixel_vec_max_width
    var fourier = ComplexImage.D1[DType.float32](sx=max_width)

    @parameter
    for i in range(max_width):
        var f: Float32 = (i + 1)*2
        fourier[x=i] = ComplexFloat32(f - 1, f)
    
    fourier.multiply(3.0)

    @parameter
    for i in range(max_width):
        var f: Float32 = (i + 1)*2
        assert_equal(fourier[x=i], ComplexFloat32((f - 1)*3, f*3))
