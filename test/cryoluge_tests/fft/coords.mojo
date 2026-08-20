
from testing import assert_equal, assert_true
from builtin._location import __call_location, _SourceLocation

from cryoluge.lang import LexicalScope
from cryoluge.image import Vec
from cryoluge.fft import FFTCoords, FFTCoordsFull


comptime funcs = __functions_in_module()


def test_coords():

    # 1D tests

    with LexicalScope():
        var sizes_real = Vec[1](x=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[1](x=3))
        assert_equal(coords.fmin(), Vec[1](x=-2))
        assert_equal(coords.fmax(), Vec[1](x=2))

        assert_equal(coords.f2i(Vec[1](x=-2)), Vec[1](x=2))
        assert_equal(coords.f2i(Vec[1](x=-1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.f2i(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=2)), Vec[1](x=2))

        assert_equal(coords.i2f(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.i2f(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.i2f(Vec[1](x=2)), Vec[1](x=2))

    with LexicalScope():
        var sizes_real = Vec[1](x=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[1](x=3))
        assert_equal(coords.fmin(), Vec[1](x=-2))
        assert_equal(coords.fmax(), Vec[1](x=2))

        assert_equal(coords.f2i(Vec[1](x=-2)), Vec[1](x=2))
        assert_equal(coords.f2i(Vec[1](x=-1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.f2i(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=2)), Vec[1](x=2))

        assert_equal(coords.i2f(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.i2f(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.i2f(Vec[1](x=2)), Vec[1](x=2))

    # 2D tests

    with LexicalScope():
        var sizes_real = Vec[2](x=5, y=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[2](x=3, y=4))
        assert_equal(coords.fmin(), Vec[2](x=-2, y=-2))
        assert_equal(coords.fmax(), Vec[2](x=2, y=1))

        assert_equal(coords.f2i(Vec[2](x=0, y=-2)), Vec[2](x=0, y=2))
        assert_equal(coords.f2i(Vec[2](x=0, y=-1)), Vec[2](x=0, y=3))
        assert_equal(coords.f2i(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.f2i(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))

        assert_equal(coords.f2i(Vec[2](x=-1, y=-2)), Vec[2](x=1, y=2))
        assert_equal(coords.f2i(Vec[2](x=-1, y=-1)), Vec[2](x=1, y=1))
        assert_equal(coords.f2i(Vec[2](x=-1, y=0)), Vec[2](x=1, y=0))
        assert_equal(coords.f2i(Vec[2](x=-1, y=1)), Vec[2](x=1, y=3))

        assert_equal(coords.i2f(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.i2f(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.i2f(Vec[2](x=0, y=2)), Vec[2](x=0, y=-2))
        assert_equal(coords.i2f(Vec[2](x=0, y=3)), Vec[2](x=0, y=-1))

    with LexicalScope():
        var sizes_real = Vec[2](x=5, y=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[2](x=3, y=5))
        assert_equal(coords.fmin(), Vec[2](x=-2, y=-2))
        assert_equal(coords.fmax(), Vec[2](x=2, y=2))

        assert_equal(coords.f2i(Vec[2](x=0, y=-2)), Vec[2](x=0, y=3))
        assert_equal(coords.f2i(Vec[2](x=0, y=-1)), Vec[2](x=0, y=4))
        assert_equal(coords.f2i(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.f2i(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.f2i(Vec[2](x=0, y=2)), Vec[2](x=0, y=2))

        assert_equal(coords.f2i(Vec[2](x=-1, y=-2)), Vec[2](x=1, y=2))
        assert_equal(coords.f2i(Vec[2](x=-1, y=-1)), Vec[2](x=1, y=1))
        assert_equal(coords.f2i(Vec[2](x=-1, y=0)), Vec[2](x=1, y=0))
        assert_equal(coords.f2i(Vec[2](x=-1, y=1)), Vec[2](x=1, y=4))
        assert_equal(coords.f2i(Vec[2](x=-1, y=2)), Vec[2](x=1, y=3))

        assert_equal(coords.i2f(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.i2f(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.i2f(Vec[2](x=0, y=2)), Vec[2](x=0, y=2))
        assert_equal(coords.i2f(Vec[2](x=0, y=3)), Vec[2](x=0, y=-2))
        assert_equal(coords.i2f(Vec[2](x=0, y=4)), Vec[2](x=0, y=-1))

    # 3D tests

    with LexicalScope():
        var sizes_real = Vec[3](x=5, y=3, z=4)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[3](x=3, y=3, z=4))
        assert_equal(coords.fmin(), Vec[3](x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), Vec[3](x=2, y=1, z=1))

        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=-2)), Vec[3](x=0, y=0, z=2))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=-1)), Vec[3](x=0, y=0, z=3))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=0)), Vec[3](x=0, y=0, z=0))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=1)), Vec[3](x=0, y=0, z=1))

        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=0)), Vec[3](x=0, y=0, z=0))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=1)), Vec[3](x=0, y=0, z=1))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=2)), Vec[3](x=0, y=0, z=-2))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=3)), Vec[3](x=0, y=0, z=-1))

    with LexicalScope():
        var sizes_real = Vec[3](x=5, y=3, z=5)
        var coords = FFTCoords(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[3](x=3, y=3, z=5))
        assert_equal(coords.fmin(), Vec[3](x=-2, y=-1, z=-2))
        assert_equal(coords.fmax(), Vec[3](x=2, y=1, z=2))

        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=-2)), Vec[3](x=0, y=0, z=3))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=-1)), Vec[3](x=0, y=0, z=4))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=0)), Vec[3](x=0, y=0, z=0))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=1)), Vec[3](x=0, y=0, z=1))
        assert_equal(coords.f2i(Vec[3](x=0, y=0, z=2)), Vec[3](x=0, y=0, z=2))

        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=0)), Vec[3](x=0, y=0, z=0))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=1)), Vec[3](x=0, y=0, z=1))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=2)), Vec[3](x=0, y=0, z=2))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=3)), Vec[3](x=0, y=0, z=-2))
        assert_equal(coords.i2f(Vec[3](x=0, y=0, z=4)), Vec[3](x=0, y=0, z=-1))


def test_coords_contiguous():

    # 2D tests

    with LexicalScope():

        # even size
        var coords = FFTCoords(Vec[2](x=6, y=6))

        # image indices by frequency indices
        #     -3 -2 -1    0  1  2  3
        # +2  31 21 11 | 05 15 25 35  +2
        # +1  32 22 12 | 04 14 24 34  +1
        #  0  33 23 13 | 03 13 23 33   0
        #     ---------o------------
        # -1  34 24 14 | 02 12 22 32  -1
        # -2  35 25 15 | 01 11 21 31  -2
        # -3  30 20 10 | 00 10 20 30  -3
        #     -3 -2 -1    0  1  2  3

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=0)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=3)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-3)), Vec[2](x=3, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-2)), Vec[2](x=3, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-1)), Vec[2](x=3, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=0)), Vec[2](x=3, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=1)), Vec[2](x=3, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=2)), Vec[2](x=3, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=3)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-3)), Vec[2](x=1, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-2)), Vec[2](x=1, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-1)), Vec[2](x=1, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=0)), Vec[2](x=1, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=1)), Vec[2](x=1, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=2)), Vec[2](x=1, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=3)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-3)), Vec[2](x=0, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-2)), Vec[2](x=0, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-1)), Vec[2](x=0, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=0)), Vec[2](x=0, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=1)), Vec[2](x=0, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=2)), Vec[2](x=0, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=3)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-3)), Vec[2](x=3, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-2)), Vec[2](x=3, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-1)), Vec[2](x=3, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=0)), Vec[2](x=3, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=1)), Vec[2](x=3, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=2)), Vec[2](x=3, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=3)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=0)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=3)), None)

    with LexicalScope():

        # odd size
        var coords = FFTCoords(Vec[2](x=7, y=7))

        # image indices by frequency indices
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

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=-1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=0)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-4, y=4)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-3)), Vec[2](x=3, y=6))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-2)), Vec[2](x=3, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=-1)), Vec[2](x=3, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=0)), Vec[2](x=3, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=1)), Vec[2](x=3, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=2)), Vec[2](x=3, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=3)), Vec[2](x=3, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-3, y=4)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-3)), Vec[2](x=1, y=6))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-2)), Vec[2](x=1, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=-1)), Vec[2](x=1, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=0)), Vec[2](x=1, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=1)), Vec[2](x=1, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=2)), Vec[2](x=1, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=3)), Vec[2](x=1, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=-1, y=4)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-3)), Vec[2](x=0, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-2)), Vec[2](x=0, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=-1)), Vec[2](x=0, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=0)), Vec[2](x=0, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=1)), Vec[2](x=0, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=2)), Vec[2](x=0, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=3)), Vec[2](x=0, y=6))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=0, y=4)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-3)), Vec[2](x=3, y=0))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-2)), Vec[2](x=3, y=1))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=-1)), Vec[2](x=3, y=2))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=0)), Vec[2](x=3, y=3))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=1)), Vec[2](x=3, y=4))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=2)), Vec[2](x=3, y=5))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=3)), Vec[2](x=3, y=6))
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=3, y=4)), None)

        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-4)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=-1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=0)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=1)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=2)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=3)), None)
        assert_i(coords.maybe_f2i_contiguous(Vec[2](x=4, y=4)), None)


def test_coords_full():

    # 1D tests

    with LexicalScope():
        var sizes_real = Vec[1](x=4)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[1](x=5))
        assert_equal(coords.fmin(), Vec[1](x=-2))
        assert_equal(coords.fmax(), Vec[1](x=2))

        assert_equal(coords.f2i(Vec[1](x=-2)), Vec[1](x=3))
        assert_equal(coords.f2i(Vec[1](x=-1)), Vec[1](x=4))
        assert_equal(coords.f2i(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.f2i(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=2)), Vec[1](x=2))

        assert_equal(coords.i2f(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.i2f(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.i2f(Vec[1](x=2)), Vec[1](x=2))

    with LexicalScope():
        var sizes_real = Vec[1](x=5)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[1](x=5))
        assert_equal(coords.fmin(), Vec[1](x=-2))
        assert_equal(coords.fmax(), Vec[1](x=2))

        assert_equal(coords.f2i(Vec[1](x=-2)), Vec[1](x=3))
        assert_equal(coords.f2i(Vec[1](x=-1)), Vec[1](x=4))
        assert_equal(coords.f2i(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.f2i(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.f2i(Vec[1](x=2)), Vec[1](x=2))

        assert_equal(coords.i2f(Vec[1](x=0)), Vec[1](x=0))
        assert_equal(coords.i2f(Vec[1](x=1)), Vec[1](x=1))
        assert_equal(coords.i2f(Vec[1](x=2)), Vec[1](x=2))

    # 2D tests

    with LexicalScope():
        var sizes_real = Vec[2](x=5, y=4)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[2](x=5, y=4))
        assert_equal(coords.fmin(), Vec[2](x=-2, y=-2))
        assert_equal(coords.fmax(), Vec[2](x=2, y=1))

        assert_equal(coords.f2i(Vec[2](x=0, y=-2)), Vec[2](x=0, y=2))
        assert_equal(coords.f2i(Vec[2](x=0, y=-1)), Vec[2](x=0, y=3))
        assert_equal(coords.f2i(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.f2i(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))

        assert_equal(coords.i2f(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.i2f(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.i2f(Vec[2](x=0, y=2)), Vec[2](x=0, y=-2))
        assert_equal(coords.i2f(Vec[2](x=0, y=3)), Vec[2](x=0, y=-1))

    with LexicalScope():
        var sizes_real = Vec[2](x=5, y=5)
        var coords = FFTCoordsFull(sizes_real)

        assert_equal(coords.sizes_real(), sizes_real)
        assert_equal(coords.sizes_fourier(), Vec[2](x=5, y=5))
        assert_equal(coords.fmin(), Vec[2](x=-2, y=-2))
        assert_equal(coords.fmax(), Vec[2](x=2, y=2))

        assert_equal(coords.f2i(Vec[2](x=0, y=-2)), Vec[2](x=0, y=3))
        assert_equal(coords.f2i(Vec[2](x=0, y=-1)), Vec[2](x=0, y=4))
        assert_equal(coords.f2i(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.f2i(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.f2i(Vec[2](x=0, y=2)), Vec[2](x=0, y=2))

        assert_equal(coords.i2f(Vec[2](x=0, y=0)), Vec[2](x=0, y=0))
        assert_equal(coords.i2f(Vec[2](x=0, y=1)), Vec[2](x=0, y=1))
        assert_equal(coords.i2f(Vec[2](x=0, y=2)), Vec[2](x=0, y=2))
        assert_equal(coords.i2f(Vec[2](x=0, y=3)), Vec[2](x=0, y=-2))
        assert_equal(coords.i2f(Vec[2](x=0, y=4)), Vec[2](x=0, y=-1))

    # NOTE: 3D is pretty much the same as 2D


@always_inline
def assert_i[dim: Int, //](
    obs: Optional[Vec[dim,Int]],
    exp: Optional[Vec[dim,Int]],
    *,
    location: Optional[_SourceLocation] = None
):
    var matches: Bool
    if obs is None:
        if exp is None:
            matches = True
        else:
            matches = False
    else:
        if exp is None:
            matches = False
        else:
            matches = obs.value() == exp.value()

    assert_true(
        matches,
        String("Indices mismatch!\n",
            "\tobserved: ", String(obs.value()) if obs is not None else "(none)", "\n",
            "\texpected: ", String(exp.value()) if exp is not None else "(none)"
        ),
        location=location.or_else(__call_location())
    )
