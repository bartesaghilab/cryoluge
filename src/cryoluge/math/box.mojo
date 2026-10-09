
from cryoluge.math.vec import Vec
from cryoluge.math.matrix import Matrix


struct AlignedBox[
    dim: Int,
    dtype: DType
](
    Copyable,
    Movable
) where dim > 0:

    """Upper bounds are exclusive."""
    # TODO: parameterize bounds behavior?
    var origin: Self.Vec
    var sizes: Self.Vec

    comptime Vec = Vec[Self.dim,Self.dtype]

    @staticmethod
    def unit_corners(out corners: List[Self.Vec]):
        """
        Return a list of vectors pointing to each corner of the unit box.
        """

        # start with the zero vec
        corners = [Self.Vec(fill=0)]

        # add the other deltas by flipping 0s to 1s
        comptime for d in range(Self.dim):
            for i in range(len(corners)):
                var corner = corners[i].copy()
                corner[d] = 1
                corners.append(corner^)

    def __init__(out self, *, origin: Self.Vec, sizes: Self.Vec):
        self.origin = origin.copy()
        self.sizes = sizes.copy()

    def __init__(out self, *, min: Self.Vec, max: Self.Vec):
        """Max is exclusive."""
        self = Self(
            origin = min.copy(),
            sizes = max - min
        )

    def corner(self, dir: Self.Vec) -> Self.Vec:
        return self.origin + dir*self.sizes

    def max(self, out max: Self.Vec):
        max = self.origin + self.sizes


struct OrientedBox[
    dim: Int,
    dtype: DType
](
    Copyable,
    Movable
) where dim > 0:

    var aligned: AlignedBox[Self.dim,Self.dtype]
    var orientation: Matrix[Self.dim,Self.dim,Self.dtype]
    """A rotation matrix."""

    comptime Mat = Matrix[Self.dim,Self.dim,Self.dtype]
    comptime Vec = Vec[Self.dim,Self.dtype]

    def __init__(
        out self,
        *,
        origin: Self.Vec,
        sizes: Self.Vec,
        orientation: Matrix[Self.dim,Self.dim,Self.dtype]
    ):
        self.aligned = AlignedBox(origin=origin, sizes=sizes)
        self.orientation = orientation.copy()

    def sizes(ref self) -> ref [origin_of(self.aligned.sizes)] Self.Vec:
        return self.aligned.sizes

    def corner(self, dir: Self.Vec) -> Self.Vec:
        var scaled_dir = dir*self.aligned.sizes
        return self.aligned.origin + self.orientation*scaled_dir

    def bounding_box(self) -> AlignedBox[Self.dim,Self.dtype]:

        # start with the origin
        var min = self.aligned.origin.copy()
        var max = self.aligned.origin.copy()
        
        # iterate over all the corners of the box to find the min and max extents
        comptime for dir in AlignedBox[Self.dim,Self.dtype].unit_corners():
            var corner = self.corner(materialize[dir]())
            min = min.min(corner)
            max = max.max(corner)

        return AlignedBox(min=min, max=max)
