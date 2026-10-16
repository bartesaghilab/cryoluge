
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
    Movable
):
    var _sizes: Vec[Self.dim,.int]

    def __init__(out self, sizes: Vec[Self.dim,.int]):
        self._sizes = sizes.copy()

    def iterate(
        self,
        func: Some[def (i: Vec[Self.dim,.int])]
    ):
        
        comptime if Self.dim == 1:
            
            for x in range(self._sizes.x()):
                func([x])

        elif Self.dim == 2:
            
            for y in range(self._sizes.y()):
                for x in range(self._sizes.x()):
                    func([x, y])

        elif Self.dim == 3:

            for z in range(self._sizes.z()):
                for y in range(self._sizes.y()):
                    for x in range(self._sizes.x()):
                        func([x, y, z])

        else:
            unimplemented_dimension[Self.dim]()
