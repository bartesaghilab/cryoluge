
from std.testing import assert_true, assert_false

from cryoluge_testlib import TestSuite

from . import euler
from . import vec
from . import matrix
from . import rand
from . import box
from . import units


from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_is_even() raises:

    from cryoluge.math import is_even

    assert_true(is_even(4))
    assert_false(is_even(3))
