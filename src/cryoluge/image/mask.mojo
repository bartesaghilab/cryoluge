
from os import abort
from math import pi, sqrt

from cryoluge.math import Dimension, Vec, sinc, clamp
from cryoluge.fft import FFTCoords, FFTImage, CoordDomain, constrain_coord_domain


# TODO: refactor to use px units?


trait MaskReal:
    fn includes[
        dim: Dimension
    ](self, i: Vec[Int,dim], sizes: Vec[Int,dim]) -> Bool: ...

trait MaskFourier:
    fn includes[
        dim: Dimension,
        origin: Origin[mut=False]
    ](self, i: Vec[Int,dim], fft_coords: FFTCoords[dim,origin]) -> Bool: ...


struct AllMask(MaskReal, MaskFourier):

    fn __init__(out self):
        pass

    fn includes[
        dim: Dimension
    ](self, i: Vec[Int,dim], sizes: Vec[Int,dim]) -> Bool:
        return True

    fn includes[
        dim: Dimension,
        origin: Origin[mut=False]
    ](self, i: Vec[Int,dim], fft_coords: FFTCoords[dim,origin]) -> Bool:
        return True


@fieldwise_init
struct MaskRegion(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    
    comptime Inside = MaskRegion(0)
    comptime Outside = MaskRegion(1)

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.Inside:
            writer.write("Inside")
        elif self == Self.Outside:
            writer.write("Outside")
        else:
            writer.write("(unrecognized MaskRegion value: ", self.value, ")")

    fn __str__(self) -> String:
        return String.write(self)


fn unrecognized_mask_region[region: MaskRegion, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Unsupported MaskRegion: ", region)]()
    return abort[T]()


fn center_dist2_real[
    dim: Dimension,
    dtype: DType
](
    i: Vec[Int,dim],
    sizes: Vec[Int,dim],
    out dist2: Scalar[dtype]
):
    """Calculate squared distance from the center, in real coordinates."""
    var pos = i.map_scalar[dtype]()
    var center = sizes.map_scalar[dtype]()/Scalar[dtype](2)
    dist2 = (pos - center).len2()


fn center_dist2_fourier[
    dim: Dimension,
    dtype: DType
](
    i: Vec[Int,dim],
    fft_coords: FFTCoords[dim],
    out dist2: Scalar[dtype]
):
    """Calculate squared distance from the center, in Fourier coordinates."""
    var freq = fft_coords.i2f(i).map_scalar[dtype]()
    var sizes = fft_coords.sizes_real().map_scalar[dtype]()
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
    var radius: Scalar[dtype]
    var _r2: Scalar[dtype]

    comptime RealInsideExclusive = RadialMask[CoordDomain.Real, MaskRegion.Inside, False, _]
    comptime RealInsideInclusive = RadialMask[CoordDomain.Real, MaskRegion.Inside, True, _]
    comptime RealOutsideExclusive = RadialMask[CoordDomain.Real, MaskRegion.Outside, False, _]
    comptime RealOutsideInclusive = RadialMask[CoordDomain.Real, MaskRegion.Outside, True, _]
    comptime FourierInsideExclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Inside, False, _]
    comptime FourierInsideInclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Inside, True, _]
    comptime FourierOutsideExclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Outside, False, _]
    comptime FourierOutsideInclusive = RadialMask[CoordDomain.Fourier, MaskRegion.Outside, True, _]

    fn __init__(out self, radius: Scalar[dtype]):
        self.radius = radius
        self._r2 = radius*radius

    fn includes(self, r2: Scalar[dtype]) -> Bool:
        @parameter
        if region == MaskRegion.Inside:
            @parameter
            if include_boundary:
                return r2 <= self._r2
            else:
                return r2 < self._r2
        elif region == MaskRegion.Outside:
            @parameter
            if include_boundary:
                return r2 >= self._r2
            else:
                return r2 > self._r2
        else:
            return unrecognized_mask_region[region,Bool]()

    fn includes[
        dim: Dimension
    ](
        self,
        i: Vec[Int,dim],
        sizes: Vec[Int,dim]
    ) -> Bool:
        constrain_coord_domain["includes()", domain, CoordDomain.Real]()
        return self.includes(center_dist2_real[dim,dtype](i, sizes))

    fn includes[
        dim: Dimension,
        origin: Origin[mut=False]
    ](
        self,
        i: Vec[Int,dim],
        fft_coords: FFTCoords[dim, origin]
    ) -> Bool:
        constrain_coord_domain["includes()", domain, CoordDomain.Fourier]()
        return self.includes(center_dist2_fourier[dim,dtype](i, fft_coords))

    # TODO: move these into a generic image ops namespace?

    fn correct_sinc[
        dim: Dimension, //
    ](
        self: RadialMask[CoordDomain.Real, region, include_boundary, DType.float32],
        mut img: Image[dim,DType.float32]
    ):
        """TODO: what does this do? Someone must know."""

        var scale = pi/img.sizes().map_float32()
        var weight_outside = sinc(scale.x()*self.radius)
        weight_outside *= weight_outside

        @parameter
        fn func(i: Vec[Int,dim]):
            if self.includes(i, img.sizes()):
                var dist = i - img.sizes()//2
                var weight = (dist.map_float32()*scale).sinc().product()
                weight *= weight
                img[i=i] /= weight
            else:
                img[i=i] /= weight_outside

        img.iterate[func]()


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
    var radius_inner: Scalar[dtype]
    var radius_outer: Scalar[dtype]
    var _r12: Scalar[dtype]
    var _r22: Scalar[dtype]

    comptime RealInsideInclusive = AnnularMask[CoordDomain.Real, MaskRegion.Inside, True, True, _]
    comptime FourierInsideInclusive = AnnularMask[CoordDomain.Fourier, MaskRegion.Inside, True, True, _]
    comptime EaseFn = fn[dtype_ease: DType, width: Int](SIMD[dtype_ease,width]) -> SIMD[dtype_ease,width]

    fn __init__(out self, radius_inner: Scalar[dtype], radius_outer: Scalar[dtype]):
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self._r12 = radius_inner*radius_inner
        self._r22 = radius_outer*radius_outer

    fn __init__(out self, *, radius_center: Scalar[dtype], width: Scalar[dtype]):
        self = Self(
            radius_inner = clamp(radius_center - width/2, min=0),
            radius_outer = (radius_center + width/2)
        )

    fn includes(self, r2: Scalar[dtype]) -> Bool:
        @parameter
        if region == MaskRegion.Inside:
            @parameter
            if include_boundary_inner and include_boundary_outer:
                return r2 >= self._r12 and r2 <= self._r22
            elif not include_boundary_inner and include_boundary_outer:
                return r2 > self._r12 and r2 <= self._r22
            elif include_boundary_inner and not include_boundary_outer:
                return r2 >= self._r12 and r2 < self._r22
            else:
                return r2 > self._r12 and r2 < self._r22
        elif region == MaskRegion.Outside:
            @parameter
            if include_boundary_inner and include_boundary_outer:
                return r2 <= self._r12 or r2 >= self._r22
            elif not include_boundary_inner and include_boundary_outer:
                return r2 < self._r12 or r2 >= self._r22
            elif include_boundary_inner and not include_boundary_outer:
                return r2 <= self._r12 or r2 > self._r22
            else:
                return r2 < self._r12 or r2 > self._r22
        else:
            constrained[False, String("Unsupported MaskRegion: ", region)]()
            return abort[Bool]()

    fn includes[
        dim: Dimension
    ](self, i: Vec[Int,dim], sizes: Vec[Int,dim]) -> Bool:
        # can't use conditional conformance (ie, bounds on self) for trait impls, so constrain explicitly
        constrain_coord_domain["includes()", domain, CoordDomain.Real]()
        return self.includes(center_dist2_real[dim,dtype](i, sizes))

    fn includes[
        dim: Dimension,
        origin: Origin[mut=False]
    ](self, i: Vec[Int,dim], fft_coords: FFTCoords[dim, origin]) -> Bool:
        # can't use conditional conformance (ie, bounds on self) for trait impls, so constrain explicitly
        constrain_coord_domain["includes()", domain, CoordDomain.Fourier]()
        return self.includes(center_dist2_fourier[dim,dtype](i, fft_coords))

    # TODO: move these into a generic image ops namespace?

    fn _blend_interpolate[
        dir: AnnularBlendDirection,
        ease: Self.EaseFn
    ](self, r2: Scalar[dtype]) -> Scalar[dtype]:
        
        var r = sqrt(r2)

        # compute the iterpolation parameter based on the direction and distance
        var t: Scalar[dtype]
        @parameter
        if dir == AnnularBlendDirection.In:
            t = self.radius_outer - r
        elif dir == AnnularBlendDirection.Out:
            t = r - self.radius_inner
        else:
            return unrecognized_mask_region[region,Scalar[dtype]]()
        t /= self.radius_outer - self.radius_inner

        # apply easing function
        return ease(t)

    fn _past_end[
        dir: AnnularBlendDirection
    ](self, r2: Scalar[dtype]) -> Bool:
        # outside the annulus: determine if we're past the end
        # (otherwise, we're before the start, but we don't need to do anything there)
        @parameter
        if dir == AnnularBlendDirection.In:
            @parameter
            if include_boundary_inner:
                return r2 < self._r12
            else:
                return r2 <= self._r12
        elif dir == AnnularBlendDirection.Out:
            @parameter
            if include_boundary_outer:
                return r2 > self._r12
            else:
                return r2 >= self._r12
        else:
            return unrecognized_mask_region[region,Bool]()

    # TODO: refactor into image operators?

    fn blend[
        dir: AnnularBlendDirection,
        *,
        ease: Self.EaseFn,
        dim: Dimension
    ](
        self: AnnularMask[CoordDomain.Real, region, include_boundary_inner, include_boundary_outer, dtype],
        mut img: Image[dim,dtype],
        v: img.PixelType
    ):
        """
        Blend the real-space image in the given direction, starting with the original image value,
        and ending at the given constant value.
        Pixels before the annulus will remain their original values.
        Pixels after the annulus will be set to the constant value.
        """

        @parameter
        fn func(i: Vec[Int,dim]):
            var r2 = center_dist2_real[dim,dtype](i, img.sizes())
            if self.includes(r2):
                var t = self._blend_interpolate[dir,ease](r2)
                img[i=i] = img[i=i]*(1 - t) + t*v
            elif self._past_end[dir](r2):
                img[i=i] = v

        img.iterate[func]()

    fn blend[
        dir: AnnularBlendDirection,
        *,
        ease: Self.EaseFn,
        dim: Dimension
    ](
        self: AnnularMask[CoordDomain.Fourier, region, include_boundary_inner, include_boundary_outer, dtype],
        mut img: FFTImage[dim,dtype],
        v: img.PixelType
    ):
        """
        Blend the Fourier-space image in the given direction, starting with the original image value,
        and ending at the given constant value.
        Pixels before the annulus will remain their original values.
        Pixels after the annulus will be set to the constant value.
        """

        @parameter
        fn func(i: Vec[Int,dim]):
            var r2 = center_dist2_fourier[dim,dtype](i, img.coords())
            if self.includes(r2):
                var t = self._blend_interpolate[dir,ease](r2)
                img.complex[i=i] = img.complex[i=i]*(1 - t) + t*v
            elif self._past_end[dir](r2):
                img.complex[i=i] = v

        img.complex.iterate[func]()


@fieldwise_init
struct AnnularBlendDirection(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    
    comptime In = AnnularBlendDirection(0)
    """Blend from the outer radius towards the inner radius."""
    comptime Out = AnnularBlendDirection(1)
    """Blend from the inner radius towards the outer radius."""

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.In:
            writer.write("In")
        elif self == Self.Out:
            writer.write("Out")
        else:
            writer.write("(unrecognized MaskRegion value: ", self.value, ")")

    fn __str__(self) -> String:
        return String.write(self)
