
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.mrc import Reader, Mode


alias TEST_FILE = "test/cryoluge/mrc/test.mrc"


def test_dimensions():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var (nx, ny, nz) = mrc_reader.size()
        assert_equal(nx, 16)
        assert_equal(ny, 16)
        assert_equal(nz, 16)


def test_read_3d_int8():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var img = mrc_reader.read_3d_int8()
        assert_equal(img.rank(), 3)
        assert_equal(img.sizes().x(), 16)
        assert_equal(img.sizes().y(), 16)
        assert_equal(img.sizes().z(), 16)

        # spot-check a few pixels
        assert_equal(img[x=0, y=0, z=0], 103)
        assert_equal(img[x=1, y=0, z=0], 100)

        assert_equal(img[x=0, y=1, z=0], 105)
        assert_equal(img[x=1, y=1, z=0], 103)

        assert_equal(img[x=0, y=0, z=1], 105)
        assert_equal(img[x=1, y=0, z=1], 103)

        assert_equal(img[x=15, y=15, z=15], 31)


def test_read_2d_int8():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        var img = mrc_reader.read_2d_int8()
        assert_equal(img.rank(), 2)
        assert_equal(img.sizes().x(), 16)
        assert_equal(img.sizes().y(), 16)

        # spot-check a few pixels
        assert_equal(img[x=0, y=0], 103)
        assert_equal(img[x=1, y=0], 100)

        # try another slice
        img = mrc_reader.read_2d_int8(z=1)
        assert_equal(img[x=0, y=0], 105)
        assert_equal(img[x=1, y=0], 103)

        # try the last slice
        img = mrc_reader.read_2d_int8(z=15)
        assert_equal(img[x=15, y=15], 31)
