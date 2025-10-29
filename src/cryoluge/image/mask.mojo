
from os import abort
from math import sqrt


trait Mask:
    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool: ...


struct AllMask(Mask):

    fn __init__(out self):
        pass

    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool:
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
    
    alias Inside = MaskRegion(0)
    alias Outside = MaskRegion(1)

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


fn center_dist2[dim: ImageDimension, dtype: DType](i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Scalar[dtype]:
    """Calculate squared distance from the center."""
    var pos = i.map_scalar[dtype]()
    var center = sizes.map_scalar[dtype]()/Scalar[dtype](2)
    return (pos - center).square()


struct RadialMask[
    region: MaskRegion,
    include_boundary: Bool,
    dtype: DType
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

    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool:
        return self.includes(center_dist2[dim,dtype](i, sizes))


struct AnnularMask[
    region: MaskRegion,
    include_boundary_inner: Bool,
    include_boundary_outer: Bool,
    dtype: DType
](
    Copyable,
    Movable,
    Mask
):
    var radius_inner: Scalar[dtype]
    var radius_outer: Scalar[dtype]
    var _r12: Scalar[dtype]
    var _r22: Scalar[dtype]

    fn __init__(out self, radius_inner: Scalar[dtype], radius_outer: Scalar[dtype]):
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self._r12 = radius_inner*radius_inner
        self._r22 = radius_outer*radius_outer

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

    fn includes[dim: ImageDimension](self, i: VecD[Int,dim], sizes: VecD[Int,dim]) -> Bool:
        return self.includes(center_dist2[dim,dtype](i, sizes))

    fn blend[
        dim: ImageDimension,
        dir: AnnularBlendDirection,
        *,
        ease: fn[dtype_ease: DType, width: Int](SIMD[dtype_ease,width]) -> SIMD[dtype_ease,width]
    ](
        self,
        mut img: Image[dim,dtype],
        v: img.PixelType
    ):
        """
        Blend the image in the given direction, starting with the original image value,
        and ending at the given constant value.
        Pixels before the annulus will remain their original values.
        Pixels after the annulus will be set to the constant value.
        """

        var r2_start: Scalar[dtype]
        @parameter
        if dir == AnnularBlendDirection.In:
            r2_start = self._r22
        elif dir == AnnularBlendDirection.Out:
            r2_start = self._r12
        else:
            return unrecognized_mask_region[region]()

        var r_start = sqrt(r2_start)

        print('hack', r_start)  # TEMP: to work around compiler bug

        @parameter
        fn func(i: VecD[Int,dim]):
            var r2 = center_dist2[dim,dtype](i, img.sizes())
            if self.includes(r2):

                var r = sqrt(r2)

                # compute the iterpolation parameter based on the direction and distance
                var t: Scalar[dtype]
                @parameter
                if dir == AnnularBlendDirection.In:
                    t = r_start - r
                elif dir == AnnularBlendDirection.Out:
                    t = r - r_start
                else:
                    return unrecognized_mask_region[region]()
                t /= self.radius_outer - self.radius_inner

                # apply easing function
                t = ease(t)

                # finally, blend the pixel
                img[i=i] = img[i=i]*(1 - t) + t*v

            else:

                # outside the annulus: determine if we're past the end
                # (otherwise, we're before the start, but we don't need to do anything there)
                var past_end: Bool
                @parameter
                if dir == AnnularBlendDirection.In:
                    @parameter
                    if include_boundary_inner:
                        past_end = r2 < self._r12
                    else:
                        past_end = r2 <= self._r12
                elif dir == AnnularBlendDirection.Out:
                    @parameter
                    if include_boundary_outer:
                        past_end = r2 > self._r12
                    else:
                        past_end = r2 >= self._r12
                else:
                    return unrecognized_mask_region[region]()

                if past_end:
                    img[i=i] = v

        img.iterate[func]()


@fieldwise_init
struct AnnularBlendDirection(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    
    alias In = AnnularBlendDirection(0)
    """Blend from the outer radius towards the inner radius."""
    alias Out = AnnularBlendDirection(1)
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
