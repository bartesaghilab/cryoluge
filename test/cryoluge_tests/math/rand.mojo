
from std.testing import assert_equal

from cryoluge.math import Rand
from cryoluge.math.error import err_abs
from cryoluge.test import assert_equal_float

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


# just some regression tests, really


comptime seed = 12345


def test_float32() raises:
    var rand = Rand(seed=seed)
    assert_equal(rand.float32(), 0.82022464)
    assert_equal(rand.float32(), 0.18554556)
    assert_equal(rand.float32(), 0.8234037)


def test_float32_avg() raises:
    var sum: Float32 = 0
    comptime N = 10_000
    var rand = Rand(seed=seed)
    for _ in range(N):
        sum += rand.float32()
    assert_equal_float(err_abs[.float32], sum/N, 0.5, eps=1e-1)


def test_float32_avg_range() raises:
    var sum: Float32 = 0
    comptime N = 10_000
    var rand = Rand(seed=seed)
    for _ in range(N):
        sum += rand.float32(min=10, max=16)
    assert_equal_float(err_abs[.float32], sum/N, 13, eps=1e-1)


def test_float64() raises:
    var rand = Rand(seed=seed)
    assert_equal(rand.float64(), 0.8202246728319548)
    assert_equal(rand.float64(), 0.8234037202520622)
    assert_equal(rand.float64(), 0.0027833676767292648)


def test_float64_avg() raises:
    var sum: Float64 = 0
    comptime N = 10_000
    var rand = Rand(seed=seed)
    for _ in range(N):
        sum += rand.float64()
    assert_equal_float(err_abs[.float64], sum/N, 0.5, eps=1e-1)


def test_float64_avg_range() raises:
    var sum: Float64 = 0
    comptime N = 10_000
    var rand = Rand(seed=seed)
    for _ in range(N):
        sum += rand.float64(min=16, max=20)
    assert_equal_float(err_abs[.float64], sum/N, 18, eps=1e-1)
