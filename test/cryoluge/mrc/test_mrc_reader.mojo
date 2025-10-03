
from testing import assert_equal

from cryoluge.io import FileReader, Endian
from cryoluge.mrc import Reader


alias TEST_FILE = "test/cryoluge/mrc/test.mrc"


def test_endian():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var mrc_reader = Reader(file_reader)

        assert_equal(mrc_reader.endian(), Endian.Little)
