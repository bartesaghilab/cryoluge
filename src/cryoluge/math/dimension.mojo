
from os import abort


@fieldwise_init
struct Dimension(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var rank: Int
    var name: StaticString

    # NOTE: mojo's grammar doesn't allow `1D`, `2D`, etc identifiers
    alias D1 = Self(1, "D1")
    alias D2 = Self(2, "D2")
    alias D3 = Self(3, "D3")

    fn __eq__(self, rhs: Self) -> Bool:
        return self.rank == rhs.rank

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)


fn unrecognized_dimension[dim: Dimension, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()


fn unimplemented_dimension[dim: Dimension, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Dimension not implemented yet: ", dim)]()
    return abort[T]()


fn expect_at_least_rank[dim: Dimension, rank: Int]():
    constrained[
        dim.rank >= rank,
        String("Expected dimension of at least rank ", rank, " but got ", dim)
    ]()
