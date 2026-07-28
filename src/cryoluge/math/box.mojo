
from cryoluge.math import Vec, Matrix


struct AlignedBox[
    dim: Int,
    dtype: DType
](
    Copyable,
    Movable
):
    """Upper bounds are exclusive."""
    # TODO: parameterize bounds behavior?
    var origin: Self.Vec
    var sizes: Self.Vec

    comptime Vec = Vec[dim,Scalar[dtype]]

    @staticmethod
    fn unit_corners(out corners: List[Vec[dim,Scalar[dtype]]]):
        """
        Return a list of vectors pointing to each corner of the unit box.
        """

        # start with the zero vec
        corners = [Vec[dim,Scalar[dtype]](fill=0)]

        # add the other deltas by flipping 0s to 1s
        for d in range(dim):
            for i in range(len(corners)):
                var corner = corners[i].copy()
                corner[d] = 1
                corners.append(corner^)

    fn __init__(out self, *, origin: Self.Vec, sizes: Self.Vec):
        self.origin = origin.copy()
        self.sizes = sizes.copy()

    fn __init__(out self, *, min: Self.Vec, max: Self.Vec):
        """Max is exclusive."""
        self = Self(
            origin = min.copy(),
            sizes = max - min
        )

    fn corner(self, dir: Vec[dim,Scalar[dtype]]) -> Self.Vec:
        var scaled_dir = dir*self.sizes
        return self.origin + scaled_dir

    fn max(self, out max: Self.Vec):
        max = self.origin + self.sizes


struct OrientedBox[
    dim: Int,
    dtype: DType
](
    Copyable,
    Movable
):
    var aligned: AlignedBox[dim,dtype]
    var orientation: Matrix[dim,dim,dtype]
    """A rotation matrix."""

    comptime Vec = Vec[dim,Scalar[dtype]]

    fn __init__(
        out self,
        *,
        origin: Self.Vec,
        sizes: Self.Vec,
        orientation: Matrix[dim,dim,dtype]
    ):
        self.aligned = AlignedBox(origin=origin, sizes=sizes)
        self.orientation = orientation.copy()

    fn sizes(ref self) -> ref [origin_of(self.aligned.sizes)] Self.Vec:
        return self.aligned.sizes

    fn corner(self, dir: Vec[dim,Scalar[dtype]]) -> Self.Vec:
        var scaled_dir = dir*self.aligned.sizes
        return self.aligned.origin + self.orientation*scaled_dir

    fn bounding_box(self) -> AlignedBox[dim,dtype]:

        # start with the origin
        var min = self.aligned.origin.copy()
        var max = self.aligned.origin.copy()
        
        # iterate over all the corners of the box to find the min and max extents
        @parameter
        for dir in AlignedBox[dim,dtype].unit_corners():
            var corner = self.corner(materialize[dir]())
            min = min.min(corner)
            max = max.max(corner)

        return AlignedBox(min=min, max=max)
