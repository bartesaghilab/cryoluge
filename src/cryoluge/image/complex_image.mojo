
from complex import ComplexSIMD
from algorithm import vectorize
from sys import simd_width_of

from cryoluge.io import ByteBuffer
from cryoluge.math import Dimension, ComplexScalar


struct ComplexImage[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    comptime D1 = ComplexImage[Dimension.D1,_]
    comptime D2 = ComplexImage[Dimension.D2,_]
    comptime D3 = ComplexImage[Dimension.D3,_]
    comptime Vec = Vec[_,dim]
    comptime PixelType = ComplexScalar[dtype]
    comptime PixelVec = ComplexSIMD[dtype,_]
    comptime ScalarType = Scalar[dtype]
    comptime ScalarVec = SIMD[dtype,_]
    comptime pixel_vec_max_width = simd_width_of[dtype]()

    fn __init__(out self, sizes: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes, alignment=alignment)
        # NOTE: This implementation uses an interleaved ordering for complex components.
        #       ie, Array-of-Structures (AoS):
        #         real_0, imag_0, real_1, imag_1, ...
        #       The other option is to use the Structure-of-Arrays (SoA) layout:
        #         real_0, real_1, ..., imag_0, imag_1, ...
        #       When thinking about SIMD operations, the different memory layouts have different performance tradeoffs.
        #       In both layouts, you need two load instructions to populate the two vector registers:
        #       One vector register for the real values, and one vector register for the imaginary values.
        #       In SoA layouts:
        #         The two loads will be adjacent in memory, making better use of (eg, L1, L2) cache.
        #         But you'll load the vector registers with interleaved data,
        #         so more vector intructions are needed
        #         to de-interleave the data into the pure real and imaginary components.
        #       In AoS layouts:
        #         The two loads will be addresses that could be far apart,
        #         which grealy reduces the effectiveness of cache.
        #         But no extra are instructions are needed to build the real and imaginary vector registers.
        #       The hypothesis is that two distant loads cost more than a few extra vector instructions,
        #       so AoS layout works best here.
        #       But, as with all things HPC, we should benchmark and be sure.
        #       For what it's worth, fftw uses the AoS layout, so that's probably good enough for us too.

    fn __init__(out self, *, sx: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx), alignment=alignment)

    fn __init__(out self, *, sx: Int, sy: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx, y=sy), alignment=alignment)

    fn __init__(out self, *, sx: Int, sy: Int, sz: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx, y=sy, z=sz), alignment=alignment)

    fn rank(self) -> Int:
        return self._buf.rank()

    fn num_pixels(self) -> Int:
        return self._buf.num_elements()

    fn sizes(self) -> ref [origin_of(self._buf._sizes)] Self.Vec[Int]:
        return self._buf.sizes()

    fn span(ref self, *, start: Int = 0) -> Span[Self.PixelType, origin_of(self._buf._buf)]:
        return self._buf.span(start=start)

    fn span_bytes(ref self) -> Span[Byte, origin_of(self._buf._buf)]:
        return self._buf.span_bytes()

    fn alignment(self) -> Int:
        return self._buf.alignment()

    fn __getitem__(self, *, i: Int, out v: Self.PixelType):
        v = self._buf[i=i]

    fn __getitem__(self, i: Self.Vec[Int], out v: Self.PixelType):
        v = self._buf[i]

    fn __getitem__(self, *, x: Int, out v: Self.PixelType):
        v = self._buf[x=x]

    fn __getitem__(self, *, x: Int, y: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y]

    fn __getitem__(self, *, x: Int, y: Int, z: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y, z=z]

    fn __setitem__(mut self, *, i: Int, v: Self.PixelType):
        self._buf[i=i] = v

    fn __setitem__(mut self, i: Self.Vec[Int], v: Self.PixelType):
        self._buf[i] = v

    fn __setitem__(mut self, *, x: Int, v: Self.PixelType):
        self._buf[x=x] = v

    fn __setitem__(mut self, *, x: Int, y: Int, v: Self.PixelType):
        self._buf[x=x, y=y] = v

    fn __setitem__(mut self, *, x: Int, y: Int, z: Int, v: Self.PixelType):
        self._buf[x=x, y=y, z=z] = v

    fn get(self, i: Self.Vec[Int]) -> Optional[Self.PixelType]:
        return self._buf.get(i)
    
    fn iterate[
        func: fn (i: Self.Vec[Int]) capturing
    ](self):
        self._buf.iterate[func]()

    # TEMP: for comparing to images in other programs
    fn assert_info[samples: Int](
        self: ComplexImage[dim,DType.float32],
        msg: String,
        sizes: Self.Vec[Int],
        head: InlineArray[ComplexFloat32, samples],
        tail: InlineArray[ComplexFloat32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5
    ):
        self._buf.assert_info(msg, sizes, head, tail, hash, verbose=verbose, eps=eps)

    # TEMP
    fn assert_data[err_fn: ErrFnFloat32 = err_rel](
        mut self: ComplexImage[dim,DType.float32],
        msg: String,
        path: String,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5,
        overwrite: Bool = False
    ) raises:
        self._buf.assert_data[err_fn](msg, path, verbose=verbose, eps=eps, overwrite=overwrite)

    fn _check_vector_offset[width: Int](self, offset: Int):
        var bound = self.num_pixels() - width
        debug_assert(
            offset < bound,
            "offset ", offset, " out of range [0,", bound, ")"
        )

    fn _load[width: Int](self, offset: Int, out v: Self.PixelVec[width]):

        # load the interleaved data
        # NOTE: mojo's compiler is smart enough to handle vector operations
        #       on sizes larger than the machine limits, as long as it's a power of 2
        var interleaved = self.span(start=offset)
            .unsafe_ptr()
            .bitcast[Self.ScalarType]()
            .load[width=width*2]()

        # de-interleave
        var (real, imag) = interleaved.deinterleave()

        # build the complex vector
        v = Self.PixelVec[width](
            re = rebind[Self.ScalarVec[width]](real),
            im = rebind[Self.ScalarVec[width]](imag)
        )

    fn _store[width: Int](mut self, offset: Int, v: Self.PixelVec[width]):

        # interleave
        var interleaved = v.re.interleave(v.im)

        # store the interleaved data
        self.span(start=offset)
            .unsafe_ptr()
            .bitcast[Self.ScalarType]()
            .store[width=width*2](rebind[Self.ScalarVec[width*2]](interleaved))
        
    fn pixels_read[
        func: fn[width: Int](Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](ref self):

        @parameter
        fn loader[width: Int](offset: Int):
            var v = self._load[width](offset)
            func[width](v)

        vectorize[loader, width](self.num_pixels())

    fn pixels_read_write[
        func: fn[width: Int](mut p: Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](mut self):

        @parameter
        fn loader[width: Int](offset: Int):
            var v = self._load[width](offset)
            func[width](v)
            self._store[width](offset, v)

        vectorize[loader, width](self.num_pixels())

    # TODO: move these function to struct extension functions? when that gets released?

    fn multiply[
        width: Int = Self.pixel_vec_max_width
    ](mut self, factor: Self.ScalarType):

        @parameter
        fn func[width: Int](mut p: Self.PixelVec[width]):
            p *= factor

        self.pixels_read_write[func, width]()
