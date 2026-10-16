
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
