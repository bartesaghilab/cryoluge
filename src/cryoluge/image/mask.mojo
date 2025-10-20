
from os import abort


trait Mask:
    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool: ...


struct AllMask(Mask):

    fn __init__(out self):
        pass

    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool:
        return True


@fieldwise_init
struct MaskRegion(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var value: Int
    var include_boundary: Bool

    alias ValueInside = 0
    alias ValueOutside = 1

    @staticmethod
    fn Inside[include_boundary: Bool = False](out v: Self):
        v = Self(Self.ValueInside, include_boundary)

    @staticmethod
    fn Outside[include_boundary: Bool = False](out v: Self):
        v = Self(Self.ValueOutside, include_boundary)

    fn write_to[W: Writer](self, mut writer: W):
        var b: StaticString
        if self.include_boundary:
            b = "included"
        else:
            b = "excluded"
        if self.value == Self.ValueInside:
            writer.write("Inside[boundary=", b, "]")
        elif self.value == Self.ValueOutside:
            writer.write("Outside[boundary=", b, "]")
        else:
            writer.write("(unrecognized mask value: ", self.value, ")")

    fn __str__(self) -> String:
        return String.write(self)


struct RadialMask[
    dtype: DType,
    region: MaskRegion
](
    Copyable,
    Movable,
    Mask
):
    var radius: Scalar[dtype]
    var _r2: Scalar[dtype]

    fn __init__(out self, radius: Scalar[dtype]):
        self.radius = radius
        self._r2 = radius*radius

    fn center_dist2[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Scalar[dtype]:
        """Calculate squared distance from the center."""
        var dist2: Scalar[dtype] = 0
        @parameter
        for d in range(dim.rank):
            var dist_d: Scalar[dtype] = Scalar[dtype](i[d]) - Scalar[dtype](sizes[d])/2
            dist2 += dist_d*dist_d
        return dist2

    fn includes(self, r2: Scalar[dtype]) -> Bool:
        @parameter
        if region.value == MaskRegion.ValueInside:
            @parameter
            if region.include_boundary:
                return r2 <= self._r2
            else:
                return r2 < self._r2
        elif region.value == MaskRegion.ValueOutside:
            @parameter
            if region.include_boundary:
                return r2 >= self._r2
            else:
                return r2 > self._r2
        else:
            constrained[False, String("Unsupported mask region: ", region)]()
            return abort[Bool]()

    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool:
        return self.includes(self.center_dist2(i, sizes))
