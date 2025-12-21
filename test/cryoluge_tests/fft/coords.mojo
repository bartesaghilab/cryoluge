
from testing import assert_equal

from cryoluge.lang import LexicalScope
from cryoluge.image import Vec
from cryoluge.fft import FFTCoords, FFTCoordsFull


comptime funcs = __functions_in_module()


def test_coords():

    # 1D tests

    with LexicalScope():
        var sizes_real = Vec.D1(x=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D1(x=3))
        assert_equal(coords.fmin(), Vec.D1(x=-2))
        assert_equal(coords.fmax(), Vec.D1(x=2))

        assert_equal(coords.f2i(Vec.D1(x=-2)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=-1)), Vec.D1(x=2))
        assert_equal(coords.f2i(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.f2i(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=2)), Vec.D1(x=2))

        assert_equal(coords.i2f(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.i2f(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.i2f(Vec.D1(x=2)), Vec.D1(x=2))

    with LexicalScope():
        var sizes_real = Vec.D1(x=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D1(x=3))
        assert_equal(coords.fmin(), Vec.D1(x=-2))
        assert_equal(coords.fmax(), Vec.D1(x=2))

        assert_equal(coords.f2i(Vec.D1(x=-2)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=-1)), Vec.D1(x=2))
        assert_equal(coords.f2i(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.f2i(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=2)), Vec.D1(x=2))

        assert_equal(coords.i2f(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.i2f(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.i2f(Vec.D1(x=2)), Vec.D1(x=2))

    # 2D tests

    with LexicalScope():
        var sizes_real = Vec.D2(x=5, y=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D2(x=3, y=4))
        assert_equal(coords.fmin(), Vec.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), Vec.D2(x=2, y=1))

        assert_equal(coords.f2i(Vec.D2(x=0, y=-2)), Vec.D2(x=0, y=2))
        assert_equal(coords.f2i(Vec.D2(x=0, y=-1)), Vec.D2(x=0, y=3))
        assert_equal(coords.f2i(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.f2i(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))

        assert_equal(coords.i2f(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.i2f(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.i2f(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=-2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=3)), Vec.D2(x=0, y=-1))

    with LexicalScope():
        var sizes_real = Vec.D2(x=5, y=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D2(x=3, y=5))
        assert_equal(coords.fmin(), Vec.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), Vec.D2(x=2, y=2))

        assert_equal(coords.f2i(Vec.D2(x=0, y=-2)), Vec.D2(x=0, y=3))
        assert_equal(coords.f2i(Vec.D2(x=0, y=-1)), Vec.D2(x=0, y=4))
        assert_equal(coords.f2i(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.f2i(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.f2i(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=2))

        assert_equal(coords.i2f(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.i2f(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.i2f(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=3)), Vec.D2(x=0, y=-2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=4)), Vec.D2(x=0, y=-1))

    # 3D tests

    with LexicalScope():
        var sizes_real = Vec.D3(x=5, y=3, z=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D3(x=3, y=3, z=4))
        assert_equal(coords.fmin(), Vec.D3(x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), Vec.D3(x=2, y=1, z=1))

        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=-2)), Vec.D3(x=0, y=0, z=2))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=-1)), Vec.D3(x=0, y=0, z=3))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=0)), Vec.D3(x=0, y=0, z=0))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=1)), Vec.D3(x=0, y=0, z=1))

        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=0)), Vec.D3(x=0, y=0, z=0))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=1)), Vec.D3(x=0, y=0, z=1))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=2)), Vec.D3(x=0, y=0, z=-2))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=3)), Vec.D3(x=0, y=0, z=-1))

    with LexicalScope():
        var sizes_real = Vec.D3(x=5, y=3, z=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D3(x=3, y=3, z=5))
        assert_equal(coords.fmin(), Vec.D3(x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), Vec.D3(x=2, y=1, z=2))

        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=-2)), Vec.D3(x=0, y=0, z=3))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=-1)), Vec.D3(x=0, y=0, z=4))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=0)), Vec.D3(x=0, y=0, z=0))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=1)), Vec.D3(x=0, y=0, z=1))
        assert_equal(coords.f2i(Vec.D3(x=0, y=0, z=2)), Vec.D3(x=0, y=0, z=2))

        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=0)), Vec.D3(x=0, y=0, z=0))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=1)), Vec.D3(x=0, y=0, z=1))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=2)), Vec.D3(x=0, y=0, z=2))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=3)), Vec.D3(x=0, y=0, z=-2))
        assert_equal(coords.i2f(Vec.D3(x=0, y=0, z=4)), Vec.D3(x=0, y=0, z=-1))


def test_coords_full():

    # 1D tests

    with LexicalScope():
        var sizes_real = Vec.D1(x=4)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D1(x=5))
        assert_equal(coords.fmin(), Vec.D1(x=-2))
        assert_equal(coords.fmax(), Vec.D1(x=2))

        assert_equal(coords.f2i(Vec.D1(x=-2)), Vec.D1(x=3))
        assert_equal(coords.f2i(Vec.D1(x=-1)), Vec.D1(x=4))
        assert_equal(coords.f2i(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.f2i(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=2)), Vec.D1(x=2))

        assert_equal(coords.i2f(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.i2f(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.i2f(Vec.D1(x=2)), Vec.D1(x=2))

    with LexicalScope():
        var sizes_real = Vec.D1(x=5)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D1(x=5))
        assert_equal(coords.fmin(), Vec.D1(x=-2))
        assert_equal(coords.fmax(), Vec.D1(x=2))

        assert_equal(coords.f2i(Vec.D1(x=-2)), Vec.D1(x=3))
        assert_equal(coords.f2i(Vec.D1(x=-1)), Vec.D1(x=4))
        assert_equal(coords.f2i(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.f2i(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.f2i(Vec.D1(x=2)), Vec.D1(x=2))

        assert_equal(coords.i2f(Vec.D1(x=0)), Vec.D1(x=0))
        assert_equal(coords.i2f(Vec.D1(x=1)), Vec.D1(x=1))
        assert_equal(coords.i2f(Vec.D1(x=2)), Vec.D1(x=2))

    # 2D tests

    with LexicalScope():
        var sizes_real = Vec.D2(x=5, y=4)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D2(x=5, y=4))
        assert_equal(coords.fmin(), Vec.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), Vec.D2(x=2, y=1))

        assert_equal(coords.f2i(Vec.D2(x=0, y=-2)), Vec.D2(x=0, y=2))
        assert_equal(coords.f2i(Vec.D2(x=0, y=-1)), Vec.D2(x=0, y=3))
        assert_equal(coords.f2i(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.f2i(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))

        assert_equal(coords.i2f(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.i2f(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.i2f(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=-2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=3)), Vec.D2(x=0, y=-1))

    with LexicalScope():
        var sizes_real = Vec.D2(x=5, y=5)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec.D2(x=5, y=5))
        assert_equal(coords.fmin(), Vec.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), Vec.D2(x=2, y=2))

        assert_equal(coords.f2i(Vec.D2(x=0, y=-2)), Vec.D2(x=0, y=3))
        assert_equal(coords.f2i(Vec.D2(x=0, y=-1)), Vec.D2(x=0, y=4))
        assert_equal(coords.f2i(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.f2i(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.f2i(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=2))

        assert_equal(coords.i2f(Vec.D2(x=0, y=0)), Vec.D2(x=0, y=0))
        assert_equal(coords.i2f(Vec.D2(x=0, y=1)), Vec.D2(x=0, y=1))
        assert_equal(coords.i2f(Vec.D2(x=0, y=2)), Vec.D2(x=0, y=2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=3)), Vec.D2(x=0, y=-2))
        assert_equal(coords.i2f(Vec.D2(x=0, y=4)), Vec.D2(x=0, y=-1))

    # NOTE: 3D is pretty much the same as 2D
