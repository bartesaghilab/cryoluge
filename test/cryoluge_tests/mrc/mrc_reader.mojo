
from std.testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.image import Image
from cryoluge.mrc import Reader, Mode

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


comptime TEST_FILE = "test/cryoluge_tests/mrc/test.mrc"


def test_dimensions() raises:
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var (nx, ny, nz) = mrc_reader.size()
        assert_equal(nx, 16)
        assert_equal(ny, 16)
        assert_equal(nz, 16)


def test_read_3d_int8() raises:
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var img = Image[3,.int8](mrc_reader.size_3(), fill=0)
        mrc_reader.read_3d_int8(img)
        assert_equal(img.dimension(), 3)
        assert_equal(img.sizes().x(), 16)
        assert_equal(img.sizes().y(), 16)
        assert_equal(img.sizes().z(), 16)

        # spot-check a few pixels
        assert_equal(img[[0, 0, 0]], 103)
        assert_equal(img[[1, 0, 0]], 100)

        assert_equal(img[[0, 1, 0]], 105)
        assert_equal(img[[1, 1, 0]], 103)

        assert_equal(img[[0, 0, 1]], 105)
        assert_equal(img[[1, 0, 1]], 103)

        assert_equal(img[[15, 15, 15]], 31)


def test_read_2d_int8() raises:
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var img = Image[2,.int8](mrc_reader.size_2(), fill=0)
        mrc_reader.read_2d_int8(img)
        assert_equal(img.dimension(), 2)
        assert_equal(img.sizes().x(), 16)
        assert_equal(img.sizes().y(), 16)

        # spot-check a few pixels
        assert_equal(img[[0, 0]], 103)
        assert_equal(img[[1, 0]], 100)

        # try another slice
        mrc_reader.read_2d_int8(img, z=1)
        assert_equal(img[[0, 0]], 105)
        assert_equal(img[[1, 0]], 103)

        # try the last slice
        mrc_reader.read_2d_int8(img, z=15)
        assert_equal(img[[15, 15]], 31)
