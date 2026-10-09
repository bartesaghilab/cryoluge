
def unrecognized_dimension[dim: Int]() -> Never:
    comptime assert False, t"Unrecognized dimension: {dim}"


def unimplemented_dimension[dim: Int]() -> Never:
    comptime assert False, t"Dimension not implemented yet: {dim}"


def expect_at_least_dim[obs: Int, exp: Int]():
    comptime assert obs >= exp,
        t"Expected dimension at least {exp}, but got dimension {obs} instead"
