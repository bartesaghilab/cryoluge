
from os import abort


fn unrecognized_dimension[dim: Int, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()


fn unimplemented_dimension[dim: Int, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Dimension not implemented yet: ", dim)]()
    return abort[T]()


fn expect_at_least_rank[obs: Int, exp: Int]():
    constrained[
        obs >= exp,
        String("Expected dimension at least ", exp, " but got ", obs)
    ]()
