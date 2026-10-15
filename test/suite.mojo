
from cryoluge_testlib import TestSuite

import cryoluge_tests


def main() raises:

    print("Discovering tests ...")

    # NOTE: add the following to each testing module:
    # from cryoluge_testlib import ModuleTests, TestLocation
    # comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())

    var suite = TestSuite()

    suite.add_module[cryoluge_tests.math.tests]()
    suite.add_module[cryoluge_tests.math.euler.tests]()
    suite.add_module[cryoluge_tests.math.vec.tests]()
    suite.add_module[cryoluge_tests.math.matrix.tests]()
    suite.add_module[cryoluge_tests.math.rand.tests]()
    suite.add_module[cryoluge_tests.math.box.tests]()

    suite.add_module[cryoluge_tests.collections.keyable_set.tests]()

    suite.add_module[cryoluge_tests.io.binary_reader.tests]()
    suite.add_module[cryoluge_tests.io.binary_writer.tests]()
    suite.add_module[cryoluge_tests.io.bytes_reader.tests]()
    suite.add_module[cryoluge_tests.io.bytes_writer.tests]()
    suite.add_module[cryoluge_tests.io.file_reader.tests]()
    suite.add_module[cryoluge_tests.io.file_writer.tests]()

    suite.add_module[cryoluge_tests.cistem.cistem_reader.tests]()
    suite.add_module[cryoluge_tests.cistem.cistem_writer.tests]()
    suite.add_module[cryoluge_tests.cistem.blocked_reader.tests]()
    suite.add_module[cryoluge_tests.cistem.blocked_writer.tests]()

    # TEMP
    # try:
        
        # TODO

        # suite._register_tests[cryoluge_tests.image.complex_image.funcs]()
        # suite._register_tests[cryoluge_tests.image.resolution_statistics.funcs]()

        # suite._register_tests[cryoluge_tests.fft.coords.funcs]()
        # suite._register_tests[cryoluge_tests.fft.fft.funcs]()
        # suite._register_tests[cryoluge_tests.fft.image.funcs]()
        # suite._register_tests[cryoluge_tests.fft.interpolation.funcs]()

        # suite._register_tests[cryoluge_tests.mrc.mrc_reader.funcs]()

        # suite._register_tests[cryoluge_tests.optimization.golden_section.funcs]()
        # suite._register_tests[cryoluge_tests.optimization.brent.funcs]()
        # suite._register_tests[cryoluge_tests.optimization.ccd.funcs]()
        # suite._register_tests[cryoluge_tests.optimization.powell.funcs]()

        # suite._register_tests[cryoluge_tests.sync.mutex.funcs]()
        # suite._register_tests[cryoluge_tests.sync.affinity.funcs]()

    # except e:
    #     suite^.abandon()
    #     raise e

    print("Running tests ...")
    suite^.run()
