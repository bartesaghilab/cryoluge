
from testing import TestSuite

import cryoluge_tests


def main():

    print(" == Discovering tests ... ==")

    # NOTE: add this to the testing modules
    # comptime funcs = __functions_in_module()

    var suite = TestSuite()
    try:

        suite._register_tests[cryoluge_tests.collections.keyable_set.funcs]()
        suite._register_tests[cryoluge_tests.collections.movable_list.funcs]()
        suite._register_tests[cryoluge_tests.collections.movable_dict.funcs]()
        suite._register_tests[cryoluge_tests.collections.movable_optional.funcs]()

        suite._register_tests[cryoluge_tests.math.euler.funcs]()
        suite._register_tests[cryoluge_tests.math.matrix.funcs]()
        suite._register_tests[cryoluge_tests.math.vec.funcs]()

        suite._register_tests[cryoluge_tests.io.binary_reader.funcs]()
        suite._register_tests[cryoluge_tests.io.binary_writer.funcs]()
        suite._register_tests[cryoluge_tests.io.bytes_reader.funcs]()
        suite._register_tests[cryoluge_tests.io.bytes_writer.funcs]()
        suite._register_tests[cryoluge_tests.io.file_reader.funcs]()
        suite._register_tests[cryoluge_tests.io.file_writer.funcs]()

        suite._register_tests[cryoluge_tests.image.complex_image.funcs]()

        suite._register_tests[cryoluge_tests.cistem.cistem_reader.funcs]()
        suite._register_tests[cryoluge_tests.cistem.blocked_reader.funcs]()

        suite._register_tests[cryoluge_tests.fft.coords.funcs]()
        suite._register_tests[cryoluge_tests.fft.fft.funcs]()
        suite._register_tests[cryoluge_tests.fft.image.funcs]()
        suite._register_tests[cryoluge_tests.fft.interpolation.funcs]()

        suite._register_tests[cryoluge_tests.mrc.mrc_reader.funcs]()

    except e:
        suite^.abandon()
        raise e

    print(" == Running tests ... ==")
    suite^.run()
