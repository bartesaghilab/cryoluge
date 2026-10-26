
from std.sys import simd_width_of, align_of
from std.ffi import OwnedDLHandle, _DLCallable
from std.complex import ComplexScalar
from std.memory import ArcPointer

from cryoluge.math import Vec
from cryoluge.math.dimension import unimplemented_dimension
from cryoluge.image import Image
from cryoluge.sync import Mutex
from cryoluge.fft.image import FFTImage


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


struct _FFTDirection(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable,
    TrivialRegisterPassable
):
    var value: Int
    
    comptime R2C = Self(0)
    comptime C2R = Self(1)

    @always_inline("builtin")
    def __init__(out self, v: Int):
        self.value = v

    @always_inline("builtin")
    def __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    def write_to[W: Writer](self, mut writer: W):
        if self == Self.R2C:
            writer.write("R2C")
        elif self == Self.C2R:
            writer.write("C2R")
        else:
            writer.write("Unknown(", self.value, ")")


# TODO: sqlite approach
# def _find_func[T: TrivialRegisterPassable](h: OwnedDLHandle, name: String) raises -> T:
#     # `OwnedDLHandle.get_function` returns a `_DLCallable` type that has
#     # an origin tied to the handle.
#     # With enough caller layers, that origin becomes impossible to express in types =(
#     # instead, of calling `get_function` here, call `get_symbol`, which returns a mere pointer
#     # that we can origin cast to something just as safe (to callers), but more convienient
#     var func = h.get_symbol[T](name)
#     if func is None:
#         raise Error("FFTW funcion not found: " + name)
#     var addr: Int = Int(func.value())
#     return Pointer(to=addr).unsafe_bitcast[FT]()[]


struct _FFTWInner[dtype: DType](
    Movable
):
    var _h: OwnedDLHandle
    var _plan_r2c_1d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _plan_c2r_1d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _plan_r2c_2d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _plan_c2r_2d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _plan_r2c_3d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _plan_c2r_3d: _DLCallable[Optional[Self._PlanPtr],ImmUntrackedOrigin]
    var _exec_r2c: _DLCallable[NoneType,ImmUntrackedOrigin]
    var _exec_c2r: _DLCallable[NoneType,ImmUntrackedOrigin]
    var _destroy_plan: _DLCallable[NoneType,ImmUntrackedOrigin]
    var _mutex: Mutex[]

    comptime _PlanPtr = Pointer[NoneType,ImmUntrackedOrigin]

    def __init__(out self) raises:

        var h: OwnedDLHandle
        var prefix: StaticString
        comptime match Self.dtype:
            case .float32:
                h = OwnedDLHandle("libfftw3f.so.3")
                prefix = "fftwf"
            case .float64:
                h = OwnedDLHandle("libfftw3.so.3")
                prefix = "fftw"
            case _:
                _unsupported_data_type[Self.dtype]()
        # NOTE: subsequent loads return the same handle,
        #       so we don't need to worry about cleanup
        #  see: https://www.man7.org/linux/man-pages/man3/dlopen.3.html

        # wipe the function origins, so we can save them as struct members
        # this is inherently unsafe, so we need to make sure the library handle
        # is always available as long as the functions are available
        # at least, from the point of view of callers
        def unsafe_launder_func[T: RegisterPassable](func: _DLCallable[T,_]) -> _DLCallable[T,MutUntrackedOrigin]:
            return _DLCallable[T](
                _opaque = func._opaque,
                _lib = func._lib
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmUntrackedOrigin]()
            )
        
        comptime PlanRet = Optional[Self._PlanPtr]
        self._plan_r2c_1d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_r2c_1d"))
        self._plan_c2r_1d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_c2r_1d"))
        self._plan_r2c_2d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_r2c_2d"))
        self._plan_c2r_2d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_c2r_2d"))
        self._plan_r2c_3d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_r2c_3d"))
        self._plan_c2r_3d = unsafe_launder_func(h.get_function[PlanRet](prefix + "_plan_dft_c2r_3d"))
        self._exec_r2c = unsafe_launder_func(h.get_function(prefix + "_execute_dft_r2c"))
        self._exec_c2r = unsafe_launder_func(h.get_function(prefix + "_execute_dft_c2r"))
        self._destroy_plan = unsafe_launder_func(h.get_function(prefix + "_destroy_plan"))
        # SAFETY: all these functions are safe to use for callers,
        #         since a caller can't drop the library handle
        #         without also dropping the functions themselves
        self._h = h^
        self._mutex = Mutex()


@fieldwise_init
struct FFTW[dtype: DType](
    Movable
):
    var _inner: ArcPointer[_FFTWInner[Self.dtype]]

    def __init__(out self) raises:
        self._inner = ArcPointer(_FFTWInner[Self.dtype]())

    def plans[dim: Int](
        self,
        sizes_real: Vec[dim, .int]
    ) raises -> FFTPlans[dim,Self.dtype]
        where dim > 0:

        # allocate some temporary images to run the fftw planner
        var alignment = best_alignment[Self.dtype]
        var img_real = Image[dim,Self.dtype](sizes_real, alignment=alignment, uninitialized=True)
        var img_fourier = FFTImage[dim,Self.dtype](sizes_real, alignment=alignment, uninitialized=True)

        return self.plans(real=img_real, fourier=img_fourier)

    def plans[dim: Int](
        self,
        *,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype]
    ) raises -> FFTPlans[dim,Self.dtype]
        where dim > 0:

        # make the plans
        var plan_r2c = self._plan[dim, .R2C](real, fourier)
        if plan_r2c is None:
            raise Error("fftw failed to return a plan for the R2C direction")
        var plan_c2r = self._plan[dim, .C2R](real, fourier)
        if plan_c2r is None:
            raise Error("fftw failed to return a plan for the C2R direction")

        return FFTPlans[dim,Self.dtype](
            _inner = self._inner.copy(),
            _plan_r2c = plan_r2c.take(),
            _plan_c2r = plan_c2r.take(),
            _info_real = _ImageInfo(real),
            _info_fourier = _ImageInfo(fourier)
        )

    def _plan[dim: Int, direction: _FFTDirection](
        self,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype]
    ) -> Optional[_FFTWInner[Self.dtype]._PlanPtr]
        where dim > 0:

        ref inner = self._inner[]

        # NOTE: the FFTW planner is very *NOT* thread-safe,
        #       so we need to get a process-wide lock before using it
        # TODO: make lock wait behavior configurable?
        with inner._mutex.lock[wait_ms=1]():

            comptime if dim == 1:

                comptime if direction == .R2C:
                    return inner._plan_r2c_1d(
                        real.sizes().x(),
                        real.span().unsafe_ptr(),
                        fourier.complex.span().unsafe_ptr(),
                        _fftw_flags
                    )
                elif direction == .C2R:
                    return inner._plan_c2r_1d(
                        real.sizes().x(),
                        fourier.complex.span().unsafe_ptr(),
                        real.span().unsafe_ptr(),
                        _fftw_flags
                    )
                else:
                    _unsupported_direction[direction]()

            elif dim == 2:

                comptime if direction == .R2C:
                    return inner._plan_r2c_2d(
                        real.sizes().y(),
                        real.sizes().x(),
                        real.span().unsafe_ptr(),
                        fourier.complex.span().unsafe_ptr(),
                        _fftw_flags
                    )
                elif direction == .C2R:
                    return inner._plan_c2r_2d(
                        real.sizes().y(),
                        real.sizes().x(),
                        fourier.complex.span().unsafe_ptr(),
                        real.span().unsafe_ptr(),
                        _fftw_flags
                    )
                else:
                    _unsupported_direction[direction]()

            elif dim == 3:

                comptime if direction == .R2C:
                    return inner._plan_r2c_3d(
                        real.sizes().z(),
                        real.sizes().y(),
                        real.sizes().x(),
                        real.span().unsafe_ptr(),
                        fourier.complex.span().unsafe_ptr(),
                        _fftw_flags
                    )
                elif direction == .C2R:
                    return inner._plan_c2r_3d(
                        real.sizes().z(),
                        real.sizes().y(),
                        real.sizes().x(),
                        fourier.complex.span().unsafe_ptr(),
                        real.span().unsafe_ptr(),
                        _fftw_flags
                    )
                else:
                    _unsupported_direction[direction]()

            else:
                unimplemented_dimension[dim]()


@fieldwise_init
struct FFTPlans[
    dim: Int,
    dtype: DType
](
    Movable
) where dim > 0:

    var _inner: ArcPointer[_FFTWInner[Self.dtype]]
    var _plan_r2c: Self.PlanPtr
    var _plan_c2r: Self.PlanPtr
    var _info_real: _ImageInfo[Self.dim]
    var _info_fourier: _ImageInfo[Self.dim]

    comptime PlanPtr = _FFTWInner[Self.dtype]._PlanPtr

    def __deinit__(deinit self):
        self._inner[]._destroy_plan(self._plan_r2c)
        self._inner[]._destroy_plan(self._plan_c2r)

    def sizes_real(self) -> Vec[Self.dim,.int]:
        return self._info_real.sizes.copy()

    def sizes_fourier(self) -> Vec[Self.dim,.int]:
        return self._info_fourier.sizes.copy()

    def alloc_real(
        self,
        *,
        uninitialized: Bool,
        out img: Image[Self.dim,Self.dtype]
    ):
        img = Image[Self.dim,Self.dtype](
            self.sizes_real(),
            alignment=self._info_real.alignment,
            uninitialized=uninitialized
        )

    def alloc_real(
        self,
        *,
        fill: Image[Self.dim,Self.dtype].PixelType,
        out img: Image[Self.dim,Self.dtype]
    ):
        img = Image[Self.dim,Self.dtype](
            self.sizes_real(),
            alignment=self._info_real.alignment,
            fill=fill
        )

    def alloc_fourier(
        self,
        *,
        uninitialized: Bool,
        out img: FFTImage[Self.dim,Self.dtype]
    ):
        img = FFTImage[Self.dim,Self.dtype](
            self.sizes_fourier(),
            alignment=self._info_fourier.alignment,
            uninitialized=uninitialized
        )

    def alloc_fourier(
        self,
        *,
        fill: FFTImage[Self.dim,Self.dtype].PixelType,
        out img: FFTImage[Self.dim,Self.dtype]
    ):
        img = FFTImage[Self.dim,Self.dtype](
            self.sizes_fourier(),
            alignment=self._info_fourier.alignment,
            fill=fill
        )

    def _check_image(self, real: Image[Self.dim,Self.dtype]):
        var info_real = _ImageInfo(real)
        debug_assert(
            info_real == self._info_real,
            "Can't run FFT, real image (", info_real, ") doesn't match planned real image (", self._info_real, ")"
        )

    def _check_image(self, fourier: FFTImage[Self.dim,Self.dtype]):
        var info_fourier = _ImageInfo(fourier)
        debug_assert(
            info_fourier == self._info_fourier,
            "Can't run FFT, fourier image (", info_fourier, ") doesn't match planned fourier image (", self._info_fourier, ")"
        )

    def r2c(
        self,
        *,
        real: Image[Self.dim,Self.dtype],
        mut fourier: FFTImage[Self.dim,Self.dtype],
        rescale: Bool = True
    ):

        # check image info, so fftw won't segfault
        self._check_image(real)
        self._check_image(fourier)

        self._inner[]._exec_r2c(
            self._plan_r2c,
            real.span().unsafe_ptr(),
            fourier.complex.span().unsafe_ptr()
        )

        # rescale in the fourier domain, if desired
        if rescale:
            var factor = 1/Scalar[Self.dtype](real.num_pixels())
            fourier.complex.multiply(factor)

    def c2r(
        self,
        *,
        fourier: FFTImage[Self.dim,Self.dtype],
        mut real: Image[Self.dim,Self.dtype]
    ):
        """
        WARNING: C2R transforms will destroy the complex input!!
                This is a "feature" of fftw, and it can't be turned off. =(
                fftw will reject FFTW_PRESERVE_INPUT flags on multi-dimensional transforms!
        """

        # check image info, so fftw won't segfault
        self._check_image(real)
        self._check_image(fourier)
        
        self._inner[]._exec_c2r(
            self._plan_c2r,
            fourier.complex.span().unsafe_ptr(),
            real.span().unsafe_ptr()
        )


@inline(.always)
def _unsupported_data_type[dtype: DType]() -> Never:
    comptime assert False, String("Unsupported data type for FFT: ", dtype)


@inline(.always)
def _unsupported_direction[direction: _FFTDirection]() -> Never:
    comptime assert False, String("Unsupported direction for FFT: ", direction)


@fieldwise_init
struct _ImageInfo[
    dim: Int
](
    Copyable,
    Movable,
    Equatable,
    Writable
) where dim > 0:

    var sizes: Vec[Self.dim, .int]
    var alignment: Optional[Int]

    def __init__(out self, img: Image[Self.dim,_]):
        self = Self(img.sizes().copy(), img.alignment())

    def __init__(out self, img: FFTImage[Self.dim,_]):
        self = Self(img.complex.sizes().copy(), img.complex.alignment())

    def __eq__(self, other: Self) -> Bool:
        return self.sizes == other.sizes
            and self.alignment == other.alignment

    def write_to[W: Writer](self, mut writer: W):
        writer.write("sizes=")
        writer.write(self.sizes)
        writer.write(", alignment=")
        writer.write(self.alignment)
