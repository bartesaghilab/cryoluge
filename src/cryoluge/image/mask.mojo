
from std.os import abort
from std.math import pi, sqrt

from cryoluge.math import Vec, sinc, clamp
from cryoluge.math.dimension import DimensionalIterator
from cryoluge.fft import FFTCoords, FFTImage, CoordDomain, expect_domain
from cryoluge.image import Image


# TODO: refactor to use px units?


trait MaskReal:
    def includes[dim: Int](self, i: Vec[dim,.int], sizes: Vec[dim,.int]) -> Bool: ...

trait MaskFourier:
    def includes[dim: Int](self, i: Vec[dim,.int], fft_coords: FFTCoords[dim]) -> Bool: ...


struct AllMask(MaskReal, MaskFourier):

    def __init__(out self):
        pass

    def includes[dim: Int](self, i: Vec[dim,.int], sizes: Vec[dim,.int]) -> Bool:
        return True

    def includes[dim: Int](self, i: Vec[dim,.int], fft_coords: FFTCoords[dim]) -> Bool:
        return True


@fieldwise_init
struct MaskRegion(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable
):
    var value: Int
    
    comptime Inside = MaskRegion(0)
    comptime Outside = MaskRegion(1)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        if self == Self.Inside:
            writer.write("Inside")
        elif self == Self.Outside:
            writer.write("Outside")
        else:
            writer.write("(unrecognized MaskRegion value: ", self.value, ")")


def unsupported_mask_region[region: MaskRegion]() -> Never:
    comptime assert False,
        String("Unsupported MaskRegion: ", region)


def center_dist2_real[
    dim: Int,
    dtype: DType
](
    i: Vec[dim,.int],
    sizes: Vec[dim,.int],
    out dist2: Scalar[dtype]
):
    """Calculate squared distance from the center, in real coordinates."""
    var pos = i.map_dtype[dtype]()
    var center = sizes.map_dtype[dtype]()/Scalar[dtype](2)
    dist2 = (pos - center).len2()


def center_dist2_fourier[
    dim: Int,
    dtype: DType
](
    i: Vec[dim,.int],
    fft_coords: FFTCoords[dim],
    out dist2: Scalar[dtype]
):
    """Calculate squared distance from the center, in Fourier coordinates."""
    var freq = fft_coords.i2f(i).map_dtype[dtype]()
    var sizes = fft_coords.sizes_real().map_dtype[dtype]()
    dist2 = (freq/sizes).len2()


struct RadialMask[
    domain: CoordDomain,
    region: MaskRegion,
    include_boundary: Bool,
    dtype: DType
](
    Copyable,
    Movable,
    MaskReal,
    MaskFourier
):
    var radius: Scalar[Self.dtype]
    var _r2: Scalar[Self.dtype]

    comptime RealInsideExclusive = RadialMask[CoordDomain.Real, MaskRegion.Inside, False, _]
    comptime RealInsideInclusive = RadialMask[CoordDomain.Real, MaskRegion.Inside, True, _]
    comptime RealOutsideExclusive = RadialMask[CoordDomain.Real, MaskRegion.Outside, False, _]
    comptime RealOutsideInclusive = RadialMask[CoordDomain.Real, MaskRegion.Outside, True, _]
    comptime FourierInsideExclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Inside, False, _]
    comptime FourierInsideInclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Inside, True, _]
    comptime FourierOutsideExclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Outside, False, _]
    comptime FourierOutsideInclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Outside, True, _]

    def __init__(out self, radius: Scalar[Self.dtype]):
        self.radius = radius
        self._r2 = radius*radius

    def includes(self, r2: Scalar[Self.dtype]) -> Bool:
        comptime if Self.region == MaskRegion.Inside:
            comptime if Self.include_boundary:
                return r2 <= self._r2
            else:
                return r2 < self._r2
        elif Self.region == MaskRegion.Outside:
            comptime if Self.include_boundary:
                return r2 >= self._r2
            else:
                return r2 > self._r2
        else:
            unsupported_mask_region[Self.region]()

    def includes[dim: Int, //](
        self,
        i: Vec[dim,.int],
        sizes: Vec[dim,.int]
    ) -> Bool:
        expect_domain[CoordDomain.Real, Self.domain]()
        return self.includes(center_dist2_real[dim,Self.dtype](i, sizes))

    def includes[dim: Int, //](
        self,
        i: Vec[dim,.int],
        fft_coords: FFTCoords[dim]
    ) -> Bool:
        expect_domain[CoordDomain.Fourier, Self.domain]()
        return self.includes(center_dist2_fourier[dim,Self.dtype](i, fft_coords))

    # TODO: move these into a generic image ops namespace?

    def correct_sinc[dim: Int, //](
        self,
        mut img: Image[dim,Self.dtype]
    ) where dim > 0 and Self.dtype.is_floating_point():
        """TODO: what does this do? Someone must know."""

        expect_domain[CoordDomain.Real, Self.domain]()

        var scale = pi/img.sizes().map_dtype[Self.dtype]()
        var weight_outside = sinc(scale.x()*self.radius)
        weight_outside *= weight_outside

        def func(i: Vec[dim,.int]) {self, mut img, scale, weight_outside} :
            if self.includes(i, img.sizes()):
                var dist = i - img.sizes()//2
                var weight = (dist.map_dtype[Self.dtype]()*scale).sinc().product()
                weight *= weight
                img[i=i] /= weight
            else:
                img[i=i] /= weight_outside

        img.pixels().iterate(func)


struct AnnularMask[
    domain: CoordDomain,
    region: MaskRegion,
    include_boundary_inner: Bool,
    include_boundary_outer: Bool,
    dtype: DType
](
    Copyable,
    Movable,
    MaskReal,
    MaskFourier
):
    var radius_inner: Scalar[Self.dtype]
    var radius_outer: Scalar[Self.dtype]
    var _r12: Scalar[Self.dtype]
    var _r22: Scalar[Self.dtype]

    comptime RealInsideInclusive = AnnularMask[CoordDomain.Real, MaskRegion.Inside, True, True, _]
    comptime FourierInsideInclusive = AnnularMask[CoordDomain.Fourier, MaskRegion.Inside, True, True, _]
    comptime EaseFn = def[dtype_ease: DType, width: Int](SIMD[dtype_ease,width]) -> SIMD[dtype_ease,width]

    def __init__(out self, radius_inner: Scalar[Self.dtype], radius_outer: Scalar[Self.dtype]):
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self._r12 = radius_inner*radius_inner
        self._r22 = radius_outer*radius_outer

    def __init__(out self, *, radius_center: Scalar[Self.dtype], width: Scalar[Self.dtype]):
        self = Self(
            radius_inner = clamp(radius_center - width/2, min=0),
            radius_outer = (radius_center + width/2)
        )

    def includes(self, r2: Scalar[Self.dtype]) -> Bool:
        comptime if Self.region == MaskRegion.Inside:
            comptime if Self.include_boundary_inner and Self.include_boundary_outer:
                return r2 >= self._r12 and r2 <= self._r22
            elif not Self.include_boundary_inner and Self.include_boundary_outer:
                return r2 > self._r12 and r2 <= self._r22
            elif Self.include_boundary_inner and not Self.include_boundary_outer:
                return r2 >= self._r12 and r2 < self._r22
            else:
                return r2 > self._r12 and r2 < self._r22
        elif Self.region == MaskRegion.Outside:
            comptime if Self.include_boundary_inner and Self.include_boundary_outer:
                return r2 <= self._r12 or r2 >= self._r22
            elif not Self.include_boundary_inner and Self.include_boundary_outer:
                return r2 < self._r12 or r2 >= self._r22
            elif Self.include_boundary_inner and not Self.include_boundary_outer:
                return r2 <= self._r12 or r2 > self._r22
            else:
                return r2 < self._r12 or r2 > self._r22
        else:
            unsupported_mask_region[Self.region]()

    def includes[dim: Int](self, i: Vec[dim,.int], sizes: Vec[dim,.int]) -> Bool:
        # can't use conditional conformance (ie, bounds on self) for trait impls, so constrain explicitly
        expect_domain[CoordDomain.Real, Self.domain]()
        return self.includes(center_dist2_real[dim,Self.dtype](i, sizes))

    def includes[dim: Int](self, i: Vec[dim,.int], fft_coords: FFTCoords[dim]) -> Bool:
        # can't use conditional conformance (ie, bounds on self) for trait impls, so constrain explicitly
        expect_domain[CoordDomain.Fourier, Self.domain]()
        return self.includes(center_dist2_fourier[dim,Self.dtype](i, fft_coords))

    # TODO: move these into a generic image ops namespace?

    def _blend_interpolate[dir: AnnularBlendDirection](
        self,
        r2: Scalar[Self.dtype],
        ease: Some[Self.EaseFn]
    ) -> Scalar[Self.dtype]:
        
        var r = sqrt(r2)

        # compute the iterpolation parameter based on the direction and distance
        var t: Scalar[Self.dtype]
        comptime if dir == AnnularBlendDirection.In:
            t = self.radius_outer - r
        elif dir == AnnularBlendDirection.Out:
            t = r - self.radius_inner
        else:
            unsupported_mask_region[Self.region]()
        t /= self.radius_outer - self.radius_inner

        # apply easing function
        return ease[Self.dtype,1](t)

    def _past_end[
        dir: AnnularBlendDirection
    ](self, r2: Scalar[Self.dtype]) -> Bool:
        # outside the annulus: determine if we're past the end
        # (otherwise, we're before the start, but we don't need to do anything there)
        comptime if dir == AnnularBlendDirection.In:
            comptime if Self.include_boundary_inner:
                return r2 < self._r12
            else:
                return r2 <= self._r12
        elif dir == AnnularBlendDirection.Out:
            comptime if Self.include_boundary_outer:
                return r2 > self._r12
            else:
                return r2 >= self._r12
        else:
            unsupported_mask_region[Self.region]()

    # TODO: refactor into image operators?

    def blend[
        dim: Int,
        //,
        dir: AnnularBlendDirection
    ](
        self,
        mut img: Image[dim,Self.dtype],
        v: img.PixelType,
        *,
        ease: Some[Self.EaseFn]
    ) where dim > 0:
        """
        Blend the real-space image in the given direction, starting with the original image value,
        and ending at the given constant value.
        Pixels before the annulus will remain their original values.
        Pixels after the annulus will be set to the constant value.
        """

        expect_domain[CoordDomain.Real, Self.domain]()

        def func(i: Vec[dim,.int]) {self, mut img, v, ease}:
            var r2 = center_dist2_real[dim,Self.dtype](i, img.sizes())
            if self.includes(r2):
                var t = self._blend_interpolate[dir](r2,ease)
                img[i=i] = img[i=i]*(1 - t) + t*v
            elif self._past_end[dir](r2):
                img[i=i] = v

        img.pixels().iterate(func)

    def blend[
        dim: Int,
        //,
        dir: AnnularBlendDirection
    ](
        self,
        mut img: FFTImage[dim,Self.dtype],
        v: img.PixelType,
        *,
        ease: Some[Self.EaseFn]
    ) where dim > 0:
        """
        Blend the Fourier-space image in the given direction, starting with the original image value,
        and ending at the given constant value.
        Pixels before the annulus will remain their original values.
        Pixels after the annulus will be set to the constant value.
        """

        expect_domain[CoordDomain.Fourier, Self.domain]()

        def func(i: Vec[dim,.int]) {self, mut img, v, ease}:
            var r2 = center_dist2_fourier[dim,Self.dtype](i, img.coords())
            if self.includes(r2):
                var t = self._blend_interpolate[dir](r2, ease)
                img.complex[i=i] = img.complex[i=i]*(1 - t) + t*v
            elif self._past_end[dir](r2):
                img.complex[i=i] = v

        img.complex.pixels().iterate(func)

    def volume_retained[
        dim: Int,
        //,
        dir: AnnularBlendDirection,
    ](
        self,
        sizes: Vec[dim,.int],
        *,
        ease: Some[Self.EaseFn],
        out volume: Scalar[Self.dtype]
    ):
        """
        Returns the sum of squared pixel values retained by the mask blend.
        """

        expect_domain[CoordDomain.Real, Self.domain]()

        var sum = Float64(0)

        # TODO: there's probably an analytical solution for this?

        def func(i: Vec[dim,.int]) {self, sizes, mut sum, ease}:
            var r2 = center_dist2_real[dim,Self.dtype](i, sizes)
            if self.includes(r2):
                var t = self._blend_interpolate[dir](r2, ease)
                sum += Float64( (1 - t)**2 )
            elif self._past_end[dir.opposite()](r2):
                sum += 1

        DimensionalIterator(sizes).iterate(func)

        volume = Scalar[Self.dtype](sum)


@fieldwise_init
struct AnnularBlendDirection(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable,
    TrivialRegisterPassable
):
    var value: Int
    
    comptime In = AnnularBlendDirection(0)
    """Blend from the outer radius towards the inner radius."""
    comptime Out = AnnularBlendDirection(1)
    """Blend from the inner radius towards the outer radius."""

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        if self == Self.In:
            writer.write("In")
        elif self == Self.Out:
            writer.write("Out")
        else:
            writer.write("(unrecognized MaskRegion value: ", self.value, ")")

    def opposite(self, out other: Self):
        other = Self(self.value ^ 0x1)


@fieldwise_init
struct EdgeMask[dim_edges: Int](MaskReal):
    var edges: Vec[Self.dim_edges, .int]
    """
    Edges are encoded as a vector of integers.
      -1 in position d encodes the low edge of dimension d.
      1 in position d encodes the high edge of dimension d.
      0 in position d encodes no edge.
      If multiple positions encode edges, the mask matches the union of those edges.
    """

    def includes[
        dim: Int
    ](self, i: Vec[dim,.int], sizes: Vec[dim,.int]) -> Bool:

        comptime for d in range(min(dim, self.dim_edges)):
            if i[d] == 0 and self.edges[d] == -1:
                return True
            elif i[d] == sizes[d] - 1 and self.edges[d] == 1:
                return True

        return False
