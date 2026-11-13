
from cryoluge.math.vec import Vec


@inline(.always)
def unrecognized_dimension[dim: Int]() -> Never:
    comptime assert False, t"Unrecognized dimension: {dim}"


@inline(.always)
def unimplemented_dimension[dim: Int]() -> Never:
    comptime assert False, t"Dimension not implemented yet: {dim}"


@inline(.always)
def expect_at_least_dim[obs: Int, exp: Int]():
    comptime assert obs >= exp,
        String("Expected dimension at least ", exp, ", but got dimension ", obs, " instead")


struct DimensionalIterator[dim: Int](
    Copyable,
    IterableOwned,
    Iterator
):
    comptime Element = Vec[Self.dim,.int]
    comptime IteratorOwnedType = Self

    var _sizes: Vec[Self.dim, .int]
    var _i: Vec[Self.dim, .int]

    @inline(.always)
    def __init__(out self, sizes: Vec[Self.dim, .int]):
        self._sizes = sizes.copy()
        self._i = Vec[Self.dim, .int](uninitialized=True)
        self.reset()

    def reset(mut self):
        self._i = Vec[Self.dim](fill=0)
        self._i[0] = -1

    def sizes(self) -> ref [self._sizes] Vec[Self.dim, .int]:
        return self._sizes
    
    @inline(.always)
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:

        # no sizes? no iteration
        if self._sizes.le_any(Vec[Self.dim](fill=0)):
            raise StopIteration()

        comptime for d in range(Self.dim):

            # try to take a step
            self._i[d] += 1

            if self._i[d] < self._sizes[d]:
                # in-range: so valid step
                return self._i.copy()
            else:
                # out-of-range: reset and try to step a higher dimension
                self._i[d] = 0

        # everything out-of-range, so we're done here
        self.reset()
        raise StopIteration()
