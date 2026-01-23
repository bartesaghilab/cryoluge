
from sys import simd_width_of, align_of
from sys.ffi import OwnedDLHandle
from os import abort
from complex import ComplexFloat32

from cryoluge.math import Dimension, Vec, unrecognized_dimension
from cryoluge.image import Image


# NOTE: fftw is perhaps not fully thread-safe?
#       https://fftw.org/fftw3_doc/Thread-safety.html
#       Might need to use locking to protect fftw internals in multi-threaded settings

# fftw docs on actually computing DFTs:
# https://fftw.org/fftw3_doc/Real_002ddata-DFTs.html
# https://fftw.org/fftw3_doc/Using-Plans.html
# https://fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html
# https://fftw.org/fftw3_doc/Multi_002ddimensional-Transforms.html
# https://fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
# https://fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html


comptime _FFTW_ESTIMATE = 1 << 6
# https://github.com/FFTW/fftw3/blob/master/api/fftw3.h#L502


# pick flags for fftw
comptime _fftw_flags = _FFTW_ESTIMATE


comptime best_alignment[dtype: DType] = align_of[SIMD[dtype, simd_width_of[dtype]()]]()


@fieldwise_init
struct FFTDirection(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    
    comptime R2C = Self(0)
    comptime C2R = Self(1)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.R2C:
            writer.write("R2C")
        elif self == Self.C2R:
            writer.write("C2R")
        else:
            writer.write("Unknown(", self.value, ")")

    fn __str__(self) -> String:
        return String.write(self)


struct FFTPlan[
    dim: Dimension,
    dtype: DType,
    direction: FFTDirection
](
    Movable
):
    var _fftw: OwnedDLHandle
    var _plan: Self.PlanPtr
    var _info_real: _ImageInfo[dim]
    var _info_fourier: _ImageInfo[dim]

    comptime PlanPtr = ImmutOpaquePointer[ImmutAnyOrigin]
    #comptime PlanPtr = ImmutOpaquePointer[ImmutOrigin.external]
    # NOTE: using (Immut|Mut)Origin.external here causes a crash, but using ImmutAnyOrigin doesn't
    #       ... no idea why
    comptime ImmutRealPtr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
    comptime MutRealPtr = UnsafePointer[Scalar[dtype], MutAnyOrigin]
    comptime ImmutComplexPtr = UnsafePointer[ComplexScalar[dtype], ImmutAnyOrigin]
    comptime MutComplexPtr = UnsafePointer[ComplexScalar[dtype], MutAnyOrigin]

    comptime R2C = FFTPlan[_,_,FFTDirection.R2C]
    comptime C2R = FFTPlan[_,_,FFTDirection.C2R]
    """
    WARNING: C2R transforms will destroy the complex input!!
             This is a "feature" of fftw, and it can't be turned off. =(
             fftw will reject FFTW_PRESERVE_INPUT flags on multi-dimensional transforms!
    """
    
    fn __init__(
        out self,
        *,
        real: Image[dim,dtype],
        fourier: FFTImage[dim,dtype]
    ) raises:

        # save the image info so we can validate it later
        self._info_real = _ImageInfo(real.sizes().copy(), real.alignment())
        self._info_fourier = _ImageInfo(fourier.complex.sizes().copy(), fourier.complex.alignment())

        self._fftw = _load_fftw[dtype]()

        # create the plan
        @parameter
        if dim == Dimension.D1:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, Self.ImmutRealPtr, Self.ImmutComplexPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_1d")
                self._plan = planner(
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, Self.ImmutComplexPtr, Self.ImmutRealPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_1d")
                self._plan = planner(
                    real.sizes().x(),
                    fourier.complex.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        elif dim == Dimension.D2:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Self.ImmutRealPtr, Self.ImmutComplexPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_2d")
                self._plan = planner(
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Self.ImmutComplexPtr, Self.ImmutRealPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_2d")
                self._plan = planner(
                    real.sizes().y(),
                    real.sizes().x(),
                    fourier.complex.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        elif dim == Dimension.D3:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Int32, Self.ImmutRealPtr, Self.ImmutComplexPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_3d")
                self._plan = planner(
                    real.sizes().z(),
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Int32, Self.ImmutComplexPtr, Self.ImmutRealPtr, UInt32) -> Self.PlanPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_3d")
                self._plan = planner(
                    real.sizes().z(),
                    real.sizes().y(),
                    real.sizes().x(),
                    fourier.complex.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        else:
            return unrecognized_dimension[dim,Self]()

        # make sure we got a plan
        if not self._plan:
            raise Error("fftw did not return a plan for this transformation")

    fn __del__(deinit self):
        var destroy_plan = self._fftw.get_function[
            fn(Self.PlanPtr)
        ](_fftw_prefix[dtype]() + "_destroy_plan")
        destroy_plan(self._plan)

    fn _check_image(self, real: Image[dim,dtype]):
        var info_real = _ImageInfo(real.sizes().copy(), real.alignment())
        debug_assert(
            info_real == self._info_real,
            "Can't run FFT, real image (", info_real, ") doesn't match planned real image (", self._info_real, ")"
        )

    fn _check_image(self, fourier: FFTImage[dim,dtype]):
        var info_fourier = _ImageInfo(fourier.complex.sizes().copy(), fourier.complex.alignment())
        debug_assert(
            info_fourier == self._info_fourier,
            "Can't run FFT, fourier image (", info_fourier, ") doesn't match planned fourier image (", self._info_fourier, ")"
        )

    fn __call__(
        self: FFTPlan[dim,dtype,FFTDirection.R2C],
        *,
        real: Image[dim,dtype],
        mut fourier: FFTImage[dim,dtype],
        rescale: Bool = True
    ):

        # check image info, so fftw won't segfault
        self._check_image(real)
        self._check_image(fourier)

        var execute = self._fftw.get_function[
            fn(Self.PlanPtr, Self.ImmutRealPtr, Self.MutComplexPtr)
        ](_fftw_prefix[dtype]() + "_execute_dft_r2c")
        # TODO: cache the fn lookup?
        execute(
            self._plan,
            real.span().unsafe_ptr(),
            fourier.complex.span().unsafe_ptr()
        )

        # rescale in the fourier domain, if desired
        if rescale:
            var factor = 1/Scalar[dtype](real.num_pixels())
            fourier.complex.multiply(factor)

    fn __call__(
        self: FFTPlan[dim,dtype,FFTDirection.C2R],
        *,
        fourier: FFTImage[dim,dtype],
        mut real: Image[dim,dtype]
    ):

        # check image info, so fftw won't segfault
        self._check_image(real)
        self._check_image(fourier)
        
        var execute = self._fftw.get_function[
            fn(Self.PlanPtr, Self.ImmutComplexPtr, Self.MutRealPtr)
        ](_fftw_prefix[dtype]() + "_execute_dft_c2r")
        # TODO: cache the fn lookup?
        execute(
            self._plan,
            fourier.complex.span().unsafe_ptr(),
            real.span().unsafe_ptr()
        )


fn _unsupported_data_type[dtype: DType, T: AnyType = NoneType._mlir_type]() -> T:
    return abort[T](String("Unsupported data type for FFT: ", dtype))


fn _unrecognized_direction[direction: FFTDirection, T: AnyType = NoneType._mlir_type]() -> T:
    return abort[T](String("Unrecognized direction for FFT: ", direction))


fn _load_fftw[dtype: DType]() raises -> OwnedDLHandle:
    @parameter
    if dtype == DType.float32:
        return OwnedDLHandle("libfftw3f.so.3")
    elif dtype == DType.float64:
        return OwnedDLHandle("libfftw3.so.3")
    else:
        return _unsupported_data_type[dtype,OwnedDLHandle]()
    # NOTE: subsequent loads return the same handle,
    #       so we don't need to worry about cleanup
    #  see: https://www.man7.org/linux/man-pages/man3/dlopen.3.html


fn _fftw_prefix[dtype: DType]() -> StaticString:
    @parameter
    if dtype == DType.float32:
        return "fftwf"
    elif dtype == DType.float64:
        return "fftw"
    else:
        return _unsupported_data_type[dtype,StaticString]()


@fieldwise_init
struct _ImageInfo[
    dim: Dimension
](
    Copyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var size: Vec[Int,dim]
    var alignment: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.size == other.size
            and self.alignment == other.alignment

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("size=")
        writer.write(self.size)
        writer.write(", alignment=")
        writer.write(self.alignment)

    fn __str__(self) -> String:
        return String.write(self)


struct FFTPlans[
    dtype: DType,
    dim: Dimension
](
    Movable
):
    var alignment: Int
    var sizes_real: Vec[Int,dim]
    var sizes_fourier: Vec[Int,dim]
    var r2c: FFTPlan.R2C[dim,dtype]
    var c2r: FFTPlan.C2R[dim,dtype]
    """
    WARNING: C2R transforms will destroy the complex input!!
             This is a "feature" of fftw, and it can't be turned off. =(
             fftw will reject FFTW_PRESERVE_INPUT flags on multi-dimensional transforms.
    """

    fn __init__(out self, sizes_real: Vec[Int,dim]) raises:

        self.alignment = best_alignment[dtype]
        self.sizes_real = sizes_real.copy()

        # allocate some temporary images to run the fftw planner
        var plan_img_real = Image[dim,dtype](self.sizes_real, alignment=self.alignment)
        var plan_img_fourier = FFTImage[dim,dtype](self.sizes_real, alignment=self.alignment)

        self.sizes_fourier = plan_img_fourier.coords().sizes_fourier()

        # make the plans
        self.r2c = FFTPlan.R2C(real=plan_img_real, fourier=plan_img_fourier)
        self.c2r = FFTPlan.C2R(real=plan_img_real, fourier=plan_img_fourier)

    fn alloc_real(self, out img: Image[dim,dtype]):
        img = Image[dim,dtype](self.sizes_real, alignment=self.alignment)

    fn alloc_fourier(self, out img: FFTImage[dim,dtype]):
        img = FFTImage[dim,dtype](self.sizes_real, alignment=self.alignment)
