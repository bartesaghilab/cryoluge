
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.cistem import Reader, ColumnType, CistemParameters


def test_reader():
    with open("test/cryoluge/cistem/test.cistem", "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)
        print('cols: ', ', '.join(cistem_reader.cols()))
        print('line size: ', cistem_reader._line_size)
        raise Error("NOPE")  # TEMP
