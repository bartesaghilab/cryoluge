
from std.testing import assert_true, assert_false

from . import euler
from . import vec
from . import matrix
from . import rand
from . import box


comptime funcs = __functions_in_module()


def test_is_even() raises:

    from cryoluge.math import is_even

    assert_true(is_even(4))
    assert_false(is_even(3))
