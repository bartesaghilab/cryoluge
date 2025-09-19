
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.cistem import BlockedReader


alias TEST_FILE = "test/cryoluge/cistem/test.blocked.cistem"
# TODO: parameters in this file:


def test_default_parameters():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        #var cistem_reader = ExtendedReader(file_reader)

        # TODO: how to handle blocks?
