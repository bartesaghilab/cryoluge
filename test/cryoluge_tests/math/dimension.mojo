
from std.testing import assert_true, assert_equal
from std.reflection import SourceLocation, call_location

from cryoluge.lang import LexicalScope
from cryoluge.math import Vec
from cryoluge.math.dimension import DimensionalIterator

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


# NOTE: mojoc can crashe while evaluating `__functions_in_module` if there are other compile errors!
#       if you see that crash, look for the hidden error and fix it, as a workaround


def test_iter_1d() raises:

    var seq: List[Vec[1,.int]]

    seq = []
    _check(seq, DimensionalIterator(Vec[1](x=0)))

    seq = [Vec[1]([0])]
    _check(seq, DimensionalIterator(Vec[1](x=1)))

    seq = [Vec[1]([0]), Vec[1]([1])]
    _check(seq, DimensionalIterator(Vec[1](x=2)))

    seq = [Vec[1]([0]), Vec[1]([1]), Vec[1]([2])]
    _check(seq, DimensionalIterator(Vec[1](x=3)))


def test_iter_2d() raises:

    var seq: List[Vec[2,.int]]

    seq = []
    _check(seq, DimensionalIterator(Vec[2](x=0, y=0)))
    _check(seq, DimensionalIterator(Vec[2](x=1, y=0)))
    _check(seq, DimensionalIterator(Vec[2](x=2, y=0)))
    _check(seq, DimensionalIterator(Vec[2](x=0, y=1)))
    _check(seq, DimensionalIterator(Vec[2](x=0, y=2)))

    seq = [Vec[2]([0, 0])]
    _check(seq, DimensionalIterator(Vec[2](x=1, y=1)))

    seq = [Vec[2]([0, 0]), Vec[2]([1, 0])]
    _check(seq, DimensionalIterator(Vec[2](x=2, y=1)))

    seq = [Vec[2]([0, 0]), Vec[2]([0, 1])]
    _check(seq, DimensionalIterator(Vec[2](x=1, y=2)))

    seq = [
        Vec[2]([0, 0]),
        Vec[2]([1, 0]),
        Vec[2]([0, 1]),
        Vec[2]([1, 1])
    ]
    _check(seq, DimensionalIterator(Vec[2](x=2, y=2)))


def test_iter_3d() raises:

    var seq: List[Vec[3,.int]]

    seq = []
    _check(seq, DimensionalIterator(Vec[3](x=0, y=0, z=0)))
    _check(seq, DimensionalIterator(Vec[3](x=1, y=0, z=0)))
    _check(seq, DimensionalIterator(Vec[3](x=0, y=1, z=0)))
    _check(seq, DimensionalIterator(Vec[3](x=0, y=0, z=1)))
    _check(seq, DimensionalIterator(Vec[3](x=5, y=0, z=0)))
    _check(seq, DimensionalIterator(Vec[3](x=0, y=5, z=0)))
    _check(seq, DimensionalIterator(Vec[3](x=0, y=0, z=5)))

    seq = [Vec[3]([0, 0, 0])]
    _check(seq, DimensionalIterator(Vec[3](x=1, y=1, z=1)))

    seq = [Vec[3]([0, 0, 0]), Vec[3]([1, 0, 0]), Vec[3]([2, 0, 0])]
    _check(seq, DimensionalIterator(Vec[3](x=3, y=1, z=1)))

    seq = [Vec[3]([0, 0, 0]), Vec[3]([0, 1, 0]), Vec[3]([0, 2, 0])]
    _check(seq, DimensionalIterator(Vec[3](x=1, y=3, z=1)))

    seq = [Vec[3]([0, 0, 0]), Vec[3]([0, 0, 1]), Vec[3]([0, 0, 2])]
    _check(seq, DimensionalIterator(Vec[3](x=1, y=1, z=3)))

    seq = [
        # the curse of dimensionality is real @.@
        Vec[3]([0, 0, 0]), Vec[3]([1, 0, 0]), Vec[3]([2, 0, 0]),
        Vec[3]([0, 1, 0]), Vec[3]([1, 1, 0]), Vec[3]([2, 1, 0]),
        Vec[3]([0, 2, 0]), Vec[3]([1, 2, 0]), Vec[3]([2, 2, 0]),
        Vec[3]([0, 0, 1]), Vec[3]([1, 0, 1]), Vec[3]([2, 0, 1]),
        Vec[3]([0, 1, 1]), Vec[3]([1, 1, 1]), Vec[3]([2, 1, 1]),
        Vec[3]([0, 2, 1]), Vec[3]([1, 2, 1]), Vec[3]([2, 2, 1]),
        Vec[3]([0, 0, 2]), Vec[3]([1, 0, 2]), Vec[3]([2, 0, 2]),
        Vec[3]([0, 1, 2]), Vec[3]([1, 1, 2]), Vec[3]([2, 1, 2]),
        Vec[3]([0, 2, 2]), Vec[3]([1, 2, 2]), Vec[3]([2, 2, 2]),
    ]
    _check(seq, DimensionalIterator(Vec[3](x=3, y=3, z=3)))


@inline(.always)
def _check[dim: Int](
    seq: List[Vec[dim,.int]],
    var iter: DimensionalIterator,
    *,
    location: Optional[SourceLocation] = None
) raises:

    var i = 0
    for _vec in iter^:
        ref vec = rebind[Vec[dim,.int]](_vec)

        assert_true(
            i < len(seq),
            "too many iterations",
            location=location.or_else(call_location())
        )
        assert_equal(
            vec,
            seq[i],
            location=location.or_else(call_location())
        )
        i += 1
