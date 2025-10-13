
from sys.ffi import DLHandle
from os import abort
from complex import ComplexFloat32

from cryoluge.image import ImageDimension, VecD, Image, ComplexImage, unrecognized_dimension


# NOTE: fftw is perhaps not fully thread-safe?
#       https://fftw.org/fftw3_doc/Thread-safety.html
#       Might need to use locking to protect fftw internals in multi-threaded settings

# fftw docs on actually computing DFTs:
# https://fftw.org/fftw3_doc/Real_002ddata-DFTs.html
# https://fftw.org/fftw3_doc/Using-Plans.html
# https://fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html
# https://fftw.org/fftw3_doc/Multi_002ddimensional-Transforms.html


alias FFTW_ESTIMATE = 1 << 6
# https://github.com/FFTW/fftw3/blob/master/api/fftw3.h#L502


# pick flags for fftw
alias _fftw_flags = FFTW_ESTIMATE


alias CPtr = UnsafePointer[Byte]
# for C FFI, the type of pointer doesn't matter


@fieldwise_init
struct FFTDirection(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    
    alias R2C = Self(0)
    alias C2R = Self(1)

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
    dim: ImageDimension,
    dtype: DType,
    direction: FFTDirection
]:
    var _fftw: DLHandle
    var _plan: CPtr

    alias R2C = FFTPlan[_,_,FFTDirection.R2C]
    alias C2R = FFTPlan[_,_,FFTDirection.C2R]
    
    fn __init__(out self, real: Image[dim,dtype], fourier: ComplexImage[dim,dtype]) raises:

        # validate the fourier sizes
        var sizes_fourier = sizes_real_to_fourier(real.sizes())
        if fourier.sizes() != sizes_fourier:
            raise Error("Expected fourier image to have sizes ", sizes_fourier, ", but it has sizes ", fourier.sizes(), " instead")

        self._fftw = _load_fftw[dtype]()

        # create the plan
        @parameter
        if dim == ImageDimension.D1:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_1d")
                self._plan = planner(
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_1d")
                self._plan = planner(
                    real.sizes().x(),
                    fourier.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        elif dim == ImageDimension.D2:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_2d")
                self._plan = planner(
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_2d")
                self._plan = planner(
                    real.sizes().y(),
                    real.sizes().x(),
                    fourier.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        elif dim == ImageDimension.D3:

            @parameter
            if direction == FFTDirection.R2C:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_r2c_3d")
                self._plan = planner(
                    real.sizes().z(),
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == FFTDirection.C2R:
                var planner = self._fftw.get_function[
                    fn(Int32, Int32, Int32, CPtr, CPtr, UInt32) -> CPtr
                ](_fftw_prefix[dtype]() + "_plan_dft_c2r_3d")
                self._plan = planner(
                    real.sizes().z(),
                    real.sizes().y(),
                    real.sizes().x(),
                    fourier.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                return _unrecognized_direction[direction,Self]()

        else:
            return unrecognized_dimension[dim,Self]()
        
    fn __del__(deinit self):
        var destroy_plan = self._fftw.get_function[
            fn(CPtr)
        ](_fftw_prefix[dtype]() + "_destroy_plan")
        destroy_plan(self._plan)

    fn __call__(self, real: Image[dim,dtype], mut fourier: ComplexImage[dim,dtype], *, rescale: Bool = True):
        @parameter
        if direction == FFTDirection.R2C:

            var execute = self._fftw.get_function[
                fn(CPtr, CPtr, CPtr)
            ](_fftw_prefix[dtype]() + "_execute_dft_r2c")
            execute(
                self._plan,
                real.span().unsafe_ptr(),
                fourier.span().unsafe_ptr()
            )

            # rescale in the fourier domain, if desired
            if rescale:
                var factor = 1/Scalar[dtype](real.num_pixels())
                fourier.multiply(factor)

        elif direction == FFTDirection.C2R:
            var execute = self._fftw.get_function[
                fn(CPtr, CPtr, CPtr)
            ](_fftw_prefix[dtype]() + "_execute_dft_c2r")
            execute(
                self._plan,
                fourier.span().unsafe_ptr(),
                real.span().unsafe_ptr()
            )
        else:
            _unrecognized_direction[direction]()


fn _unsupported_data_type[dtype: DType, T: AnyType = NoneType._mlir_type]() -> T:
    return abort[T](String("Unsupported data type for FFT: ", dtype))


fn _unrecognized_direction[direction: FFTDirection, T: AnyType = NoneType._mlir_type]() -> T:
    return abort[T](String("Unrecognized direction for FFT: ", direction))


fn _load_fftw[dtype: DType]() raises -> DLHandle:
    @parameter
    if dtype == DType.float32:
        return DLHandle("libfftw3f.so.3")
    elif dtype == DType.float64:
        return DLHandle("libfftw3.so.3")
    else:
        return _unsupported_data_type[dtype,DLHandle]()
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


fn sizes_real_to_fourier[dim: ImageDimension](sizes: VecD[UInt,dim]) -> VecD[UInt,dim]:
    var sizes_fourier = sizes.copy()
    sizes_fourier.x() = sizes.x()//2 + 1
    return sizes_fourier^
