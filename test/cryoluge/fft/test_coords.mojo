
from testing import assert_equal

from cryoluge.lang import LexicalScope
from cryoluge.image import VecD
from cryoluge.fft import FourierCoords


def test_coords():

    # 1D tests

    with LexicalScope():
        var sizes_real = VecD.D1(x=4)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D1(x=3))
        assert_equal(coords.pivot(), VecD.D1(x=0))
        assert_equal(coords.fmin(), VecD.D1(x=-2))
        assert_equal(coords.fmax(), VecD.D1(x=2))

        assert_equal(coords.f2i(VecD.D1(x=-2)), VecD.D1(x=1))
        assert_equal(coords.f2i(VecD.D1(x=-1)), VecD.D1(x=2))
        assert_equal(coords.f2i(VecD.D1(x=0)), VecD.D1(x=0))
        assert_equal(coords.f2i(VecD.D1(x=1)), VecD.D1(x=1))
        assert_equal(coords.f2i(VecD.D1(x=2)), VecD.D1(x=2))

        assert_equal(coords.i2f(VecD.D1(x=0)), VecD.D1(x=0))
        assert_equal(coords.i2f(VecD.D1(x=1)), VecD.D1(x=1))
        assert_equal(coords.i2f(VecD.D1(x=2)), VecD.D1(x=2))

    with LexicalScope():
        var sizes_real = VecD.D1(x=5)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D1(x=3))
        assert_equal(coords.pivot(), VecD.D1(x=0))
        assert_equal(coords.fmin(), VecD.D1(x=-2))
        assert_equal(coords.fmax(), VecD.D1(x=2))

        assert_equal(coords.f2i(VecD.D1(x=-2)), VecD.D1(x=1))
        assert_equal(coords.f2i(VecD.D1(x=-1)), VecD.D1(x=2))
        assert_equal(coords.f2i(VecD.D1(x=0)), VecD.D1(x=0))
        assert_equal(coords.f2i(VecD.D1(x=1)), VecD.D1(x=1))
        assert_equal(coords.f2i(VecD.D1(x=2)), VecD.D1(x=2))

        assert_equal(coords.i2f(VecD.D1(x=0)), VecD.D1(x=0))
        assert_equal(coords.i2f(VecD.D1(x=1)), VecD.D1(x=1))
        assert_equal(coords.i2f(VecD.D1(x=2)), VecD.D1(x=2))

    # 2D tests

    with LexicalScope():
        var sizes_real = VecD.D2(x=5, y=4)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D2(x=3, y=4))
        assert_equal(coords.pivot(), VecD.D2(x=0, y=2))
        assert_equal(coords.fmin(), VecD.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), VecD.D2(x=2, y=1))

        assert_equal(coords.f2i(VecD.D2(x=0, y=-2)), VecD.D2(x=0, y=2))
        assert_equal(coords.f2i(VecD.D2(x=0, y=-1)), VecD.D2(x=0, y=3))
        assert_equal(coords.f2i(VecD.D2(x=0, y=0)), VecD.D2(x=0, y=0))
        assert_equal(coords.f2i(VecD.D2(x=0, y=1)), VecD.D2(x=0, y=1))

        assert_equal(coords.i2f(VecD.D2(x=0, y=0)), VecD.D2(x=0, y=0))
        assert_equal(coords.i2f(VecD.D2(x=0, y=1)), VecD.D2(x=0, y=1))
        assert_equal(coords.i2f(VecD.D2(x=0, y=2)), VecD.D2(x=0, y=-2))
        assert_equal(coords.i2f(VecD.D2(x=0, y=3)), VecD.D2(x=0, y=-1))

    with LexicalScope():
        var sizes_real = VecD.D2(x=5, y=5)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D2(x=3, y=5))
        assert_equal(coords.pivot(), VecD.D2(x=0, y=3))
        assert_equal(coords.fmin(), VecD.D2(x=-2, y=-2))
        assert_equal(coords.fmax(), VecD.D2(x=2, y=2))

        assert_equal(coords.f2i(VecD.D2(x=0, y=-2)), VecD.D2(x=0, y=3))
        assert_equal(coords.f2i(VecD.D2(x=0, y=-1)), VecD.D2(x=0, y=4))
        assert_equal(coords.f2i(VecD.D2(x=0, y=0)), VecD.D2(x=0, y=0))
        assert_equal(coords.f2i(VecD.D2(x=0, y=1)), VecD.D2(x=0, y=1))
        assert_equal(coords.f2i(VecD.D2(x=0, y=2)), VecD.D2(x=0, y=2))

        assert_equal(coords.i2f(VecD.D2(x=0, y=0)), VecD.D2(x=0, y=0))
        assert_equal(coords.i2f(VecD.D2(x=0, y=1)), VecD.D2(x=0, y=1))
        assert_equal(coords.i2f(VecD.D2(x=0, y=2)), VecD.D2(x=0, y=2))
        assert_equal(coords.i2f(VecD.D2(x=0, y=3)), VecD.D2(x=0, y=-2))
        assert_equal(coords.i2f(VecD.D2(x=0, y=4)), VecD.D2(x=0, y=-1))

    # 3D tests

    with LexicalScope():
        var sizes_real = VecD.D3(x=5, y=3, z=4)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D3(x=3, y=3, z=4))
        assert_equal(coords.pivot(), VecD.D3(x=0, y=2, z=2))
        assert_equal(coords.fmin(), VecD.D3(x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), VecD.D3(x=2, y=1, z=1))

        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=-2)), VecD.D3(x=0, y=0, z=2))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=-1)), VecD.D3(x=0, y=0, z=3))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=0)), VecD.D3(x=0, y=0, z=0))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=1)), VecD.D3(x=0, y=0, z=1))

        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=0)), VecD.D3(x=0, y=0, z=0))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=1)), VecD.D3(x=0, y=0, z=1))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=2)), VecD.D3(x=0, y=0, z=-2))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=3)), VecD.D3(x=0, y=0, z=-1))

    with LexicalScope():
        var sizes_real = VecD.D3(x=5, y=3, z=5)
        var coords = FourierCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), VecD.D3(x=3, y=3, z=5))
        assert_equal(coords.pivot(), VecD.D3(x=0, y=2, z=3))
        assert_equal(coords.fmin(), VecD.D3(x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), VecD.D3(x=2, y=1, z=2))

        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=-2)), VecD.D3(x=0, y=0, z=3))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=-1)), VecD.D3(x=0, y=0, z=4))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=0)), VecD.D3(x=0, y=0, z=0))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=1)), VecD.D3(x=0, y=0, z=1))
        assert_equal(coords.f2i(VecD.D3(x=0, y=0, z=2)), VecD.D3(x=0, y=0, z=2))

        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=0)), VecD.D3(x=0, y=0, z=0))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=1)), VecD.D3(x=0, y=0, z=1))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=2)), VecD.D3(x=0, y=0, z=2))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=3)), VecD.D3(x=0, y=0, z=-2))
        assert_equal(coords.i2f(VecD.D3(x=0, y=0, z=4)), VecD.D3(x=0, y=0, z=-1))
