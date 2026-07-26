
from cryoluge.math import Dimension, Vec, Matrix


struct AlignedBox[
    dim: Dimension,
    dtype: DType
](
    Copyable,
    Movable
):
    """Upper bounds are exclusive."""
    # TODO: parameterize bounds behavior?
    var origin: Self.Vec
    var sizes: Self.Vec

    comptime Vec = Vec[Scalar[dtype],dim]

    fn __init__(out self, *, origin: Self.Vec, sizes: Self.Vec):
        self.origin = origin.copy()
        self.sizes = sizes.copy()

    fn __init__(out self, *, min: Self.Vec, max: Self.Vec):
        """Max is exclusive."""
        self = Self(
            origin = min.copy(),
            sizes = max - min
        )


struct OrientedBox[
    dim: Dimension,
    dtype: DType
](
    Copyable,
    Movable
):
    var aligned: AlignedBox[dim,dtype]
    var orientation: Matrix[dim.rank,dim.rank,dtype]
    """A rotation matrix."""

    comptime Vec = Vec[Scalar[dtype],dim]

    fn __init__(
        out self,
        *,
        origin: Self.Vec,
        sizes: Self.Vec,
        orientation: Matrix[dim.rank,dim.rank,dtype]
    ):
        self.aligned = AlignedBox(origin=origin, sizes=sizes)
        self.orientation = orientation.copy()

    fn bounding_box(self) -> AlignedBox[dim,dtype]:

        # start with the origin
        var min = self.aligned.origin.copy()
        var max = self.aligned.origin.copy()
        
        # iterate over all the corners of the box to find the min and max extents
        @parameter
        for _corner_vec in Dimension.unit_corners[dim]():
            var corner_vec = materialize[_corner_vec]().map_scalar[dtype]()
            var corner = self.aligned.origin + self.orientation*corner_vec
            min = min.min(corner)
            max = max.max(corner)

        return AlignedBox(min=min, max=max)
