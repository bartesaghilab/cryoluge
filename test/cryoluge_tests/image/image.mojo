
from std.testing import assert_equal

from cryoluge.math import Vec
from cryoluge.image import Image

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


def test_fill_4() raises:
    # should be evenly-divisible by one vector instruction, with no draining

    var img = Image[1,.float32]([4], fill=5)

    assert_equal(img[i=0], 5)
    assert_equal(img[i=1], 5)
    assert_equal(img[i=2], 5)
    assert_equal(img[i=3], 5)


def test_fill_5() raises:
    # should require one vector instruction, and drain with one scalar instruction

    var img = Image[1,.float32]([5], fill=5)

    assert_equal(img[i=0], 5)
    assert_equal(img[i=1], 5)
    assert_equal(img[i=2], 5)
    assert_equal(img[i=3], 5)
    assert_equal(img[i=4], 5)
