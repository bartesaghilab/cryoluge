
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
    suite.add_module[cryoluge_tests.math.units.tests]()

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

    suite.add_module[cryoluge_tests.image.image.tests]()
    suite.add_module[cryoluge_tests.image.complex_image.tests]()
    suite.add_module[cryoluge_tests.image.resolution_statistics.tests]()

    suite.add_module[cryoluge_tests.fft.coords.tests]()
    suite.add_module[cryoluge_tests.fft.image.tests]()
    suite.add_module[cryoluge_tests.fft.fft.tests]()
    suite.add_module[cryoluge_tests.fft.interpolation.tests]()

    suite.add_module[cryoluge_tests.sync.mutex.tests]()
    suite.add_module[cryoluge_tests.sync.affinity.tests]()

    suite.add_module[cryoluge_tests.mrc.mrc_reader.tests]()

    suite.add_module[cryoluge_tests.optimization.golden_section.tests]()
    suite.add_module[cryoluge_tests.optimization.brent.tests]()
    suite.add_module[cryoluge_tests.optimization.ccd.tests]()
    suite.add_module[cryoluge_tests.optimization.powell.tests]()

    print("Running tests ...")
    suite^.run()
