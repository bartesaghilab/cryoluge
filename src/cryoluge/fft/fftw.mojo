
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


struct FFTW[dtype: DType](
    Movable
):
    var _h: OwnedDLHandle
    var _prefix: StaticString

    def __init__(out self) raises:
        comptime if Self.dtype == DType.float32:
            self._h = OwnedDLHandle("libfftw3f.so.3")
            self._prefix = "fftwf"
        elif Self.dtype == DType.float64:
            self._h = OwnedDLHandle("libfftw3.so.3")
            self._prefix = "fftw"
        else:
            _unsupported_data_type[Self.dtype]()
        # NOTE: subsequent loads return the same handle,
        #       so we don't need to worry about cleanup
        #  see: https://www.man7.org/linux/man-pages/man3/dlopen.3.html

    def _funcs(self) raises -> _FFTWFuncs[Self.dtype, origin_of(self._h)]:
        comptime PlanRet = Optional[_FFTWFuncs[Self.dtype, origin_of(self._h)]._PlanPtr]
        return _FFTWFuncs[Self.dtype, origin_of(self._h)](
            _plan_r2c_1d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_r2c_1d"),
            _plan_c2r_1d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_c2r_1d"),
            _plan_r2c_2d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_r2c_2d"),
            _plan_c2r_2d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_c2r_2d"),
            _plan_r2c_3d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_r2c_3d"),
            _plan_c2r_3d = self._h.get_function[PlanRet](self._prefix + "_plan_dft_c2r_3d"),
            _exec_r2c = self._h.get_function(self._prefix + "_execute_dft_r2c"),
            _exec_c2r = self._h.get_function(self._prefix + "_execute_dft_c2r"),
            _destroy_plan = self._h.get_function(self._prefix + "_destroy_plan")
        )

    def plans[dim: Int](
        self,
        *,
        sizes_real: Vec[dim, .int],
        mutex: Optional[ArcPointer[Mutex[]]] = None
    ) raises -> FFTPlans[dim, Self.dtype, origin_of(self._h)]
        where dim > 0:

        return self._funcs().plans[dim](sizes_real, mutex)

    def plans[dim: Int](
        self,
        *,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype],
        mutex: Optional[ArcPointer[Mutex[]]] = None
    ) raises -> FFTPlans[dim, Self.dtype, origin_of(self._h)]
        where dim > 0:

        return self._funcs().plans[dim](real, fourier, mutex)


@fieldwise_init
struct _FFTWFuncs[dtype: DType, origin: Origin](
    ImplicitlyCopyable
):
    var _plan_r2c_1d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _plan_c2r_1d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _plan_r2c_2d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _plan_c2r_2d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _plan_r2c_3d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _plan_c2r_3d: _DLCallable[Optional[Self._PlanPtr],Self.origin]
    var _exec_r2c: _DLCallable[NoneType,Self.origin]
    var _exec_c2r: _DLCallable[NoneType,Self.origin]
    var _destroy_plan: _DLCallable[NoneType,Self.origin]

    comptime _PlanPtr = Pointer[NoneType,ImmUntrackedOrigin]

    def plans[dim: Int](
        self,
        sizes_real: Vec[dim, .int],
        mutex: Optional[ArcPointer[Mutex[]]] = None
    ) raises -> FFTPlans[dim,Self.dtype,Self.origin]
        where dim > 0:

        # allocate some temporary images to run the fftw planner
        var alignment = best_alignment[Self.dtype]
        var img_real = Image[dim,Self.dtype](sizes_real, alignment=alignment, uninitialized=True)
        var img_fourier = FFTImage[dim,Self.dtype](sizes_real, alignment=alignment, uninitialized=True)

        return self.plans(real=img_real, fourier=img_fourier, mutex=mutex)

    def plans[dim: Int](
        self,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype],
        mutex: Optional[ArcPointer[Mutex[]]] = None
    ) raises -> FFTPlans[dim,Self.dtype,Self.origin]
        where dim > 0:

        # make the plans
        var plan_r2c = self._plan[dim, .R2C](real, fourier, mutex)
        if plan_r2c is None:
            raise Error("fftw failed to return a plan for the R2C direction")
        var plan_c2r = self._plan[dim, .C2R](real, fourier, mutex)
        if plan_c2r is None:
            raise Error("fftw failed to return a plan for the C2R direction")

        return FFTPlans[dim,Self.dtype,Self.origin](
            _funcs = self,
            _plan_r2c = plan_r2c.take(),
            _plan_c2r = plan_c2r.take(),
            _info_real = _ImageInfo(real),
            _info_fourier = _ImageInfo(fourier)
        )

    def _plan[dim: Int, direction: _FFTDirection](
        self,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype],
        mutex: Optional[ArcPointer[Mutex[]]] = None
    ) -> Optional[Self._PlanPtr]
        where dim > 0:

        # NOTE: the FFTW planner is very *NOT* thread-safe,
        #       so we need to get a process-wide lock before using it
        if mutex is not None:
            # have mutex: lock it before planning
            # TODO: make lock wait behavior configurable?
            with mutex.value()[].lock[wait_ms=1]():
                return self._plan[dim,direction](real, fourier)
        else:
            # no mutex: YOLO
            return self._plan[dim,direction](real, fourier)

    def _plan[dim: Int, direction: _FFTDirection](
        self,
        real: Image[dim,Self.dtype],
        fourier: FFTImage[dim,Self.dtype]
    ) -> Optional[Self._PlanPtr]
        where dim > 0:

        comptime if dim == 1:

            comptime if direction == .R2C:
                return self._plan_r2c_1d(
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == .C2R:
                return self._plan_c2r_1d(
                    real.sizes().x(),
                    fourier.complex.span().unsafe_ptr(),
                    real.span().unsafe_ptr(),
                    _fftw_flags
                )
            else:
                _unsupported_direction[direction]()

        elif dim == 2:

            comptime if direction == .R2C:
                return self._plan_r2c_2d(
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == .C2R:
                return self._plan_c2r_2d(
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
                return self._plan_r2c_3d(
                    real.sizes().z(),
                    real.sizes().y(),
                    real.sizes().x(),
                    real.span().unsafe_ptr(),
                    fourier.complex.span().unsafe_ptr(),
                    _fftw_flags
                )
            elif direction == .C2R:
                return self._plan_c2r_3d(
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
    dtype: DType,
    origin: Origin
](
    Movable
) where dim > 0:

    var _funcs: _FFTWFuncs[Self.dtype,Self.origin]
    var _plan_r2c: Self.PlanPtr
    var _plan_c2r: Self.PlanPtr
    var _info_real: _ImageInfo[Self.dim]
    var _info_fourier: _ImageInfo[Self.dim]

    comptime PlanPtr = _FFTWFuncs[Self.dtype,Self.origin]._PlanPtr

    def __deinit__(deinit self):
        self._funcs._destroy_plan(self._plan_r2c)
        self._funcs._destroy_plan(self._plan_c2r)

    def alloc_real(
        self,
        *,
        uninitialized: Bool,
        out img: Image[Self.dim,Self.dtype]
    ):
        img = Image[Self.dim,Self.dtype](
            self._info_real.sizes,
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
            self._info_real.sizes,
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
            self._info_fourier.sizes,
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
            self._info_fourier.sizes,
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

        self._funcs._exec_r2c(
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
        
        self._funcs._exec_c2r(
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
