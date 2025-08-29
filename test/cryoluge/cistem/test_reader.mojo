
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.cistem import Reader


def test_reader():
    with open("test/cryoluge/cistem/test.cistem", "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)
        raise Error("NOPE")  # TEMP