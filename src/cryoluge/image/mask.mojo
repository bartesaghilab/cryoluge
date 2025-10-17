
from os import abort


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
    Movable
):
    var radius: Scalar[dtype]
    var _r2: Scalar[dtype]

    fn __init__(out self, radius: Scalar[dtype]):
        self.radius = radius
        self._r2 = radius*radius

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

    fn mean_variance[
        dim: ImageDimension, //
    ](self: Self, img: Image[dim,DType.float32]) -> (Float32, Float32):
        """Returns mean and variance, in single precision."""

        # can't seem to specialize fn impl on only some of the self parameters,
        # so make sure the types match
        constrained[
            self.dtype == DType.float32,
            String("For float32 image, mask should have dtype=float, instead has dtype=", dtype)
        ]()

        # since we can't divide by zero
        if img.num_pixels() <= 0:
            return (0, 0)

        var sum: Float64 = 0
        var sum_of_squares: Float64 = 0
        var num_pixels: Int = 0

        @parameter
        fn func(i: VecD[Int,dim]):

            # calculate squared distance from the center
            var dist2: Float32 = 0
            @parameter
            for d in range(dim.rank):
                var dist_d: Float32 = Float32(i[d]) - Float32(img.sizes()[d]/2)
                # TEMP
                if i.x() == 0 and i.y() == 0:
                    print('d[', d, '] = ', Float32(i[d]), ' - ', Float32(img.sizes()[d]/2))
                dist2 += dist_d*dist_d

            # TEMP
            if i.x() == 0 and i.y() == 0:
                print('i=', i, '  dist2=', dist2, '  r2=', self._r2)

            var dist2_scalar = rebind[Scalar[dtype]](dist2)

            if self.includes(dist2_scalar):
                num_pixels += 1
                var p = Float64(img[i])
                sum += p
                p *= p
                sum_of_squares += p

        img.iterate[func]()

        print('sum=', sum, ', sum_of_squares=', sum_of_squares, ', num_pixels', num_pixels)  # TEMP

        var n = Float64(num_pixels)
        var mean = Float32(sum/n)
        var variance = abs(Float32( sum_of_squares/n - (sum/n)*(sum/n) ))
        return (mean, variance)
