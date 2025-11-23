
from cryoluge.math import Dimension, Vec
from cryoluge.io import ByteBuffer


struct Image[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    comptime D1 = Image[Dimension.D1,_]
    comptime D2 = Image[Dimension.D2,_]
    comptime D3 = Image[Dimension.D3,_]
    comptime Vec = Vec[_,dim]
    comptime PixelType = Scalar[dtype]
    comptime PixelVec = SIMD[dtype,_]
    comptime pixel_vec_max_width = simd_width_of[dtype]()

    fn __init__(out self, sizes: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes, alignment=alignment)

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

    fn span_bytes(ref self) -> Span[Byte, origin_of(self._buf._buf)]:
        return self._buf.span_bytes()

    fn span(ref self, *, start: Int = 0) -> Span[Self.PixelType, origin_of(self._buf._buf)]:
        return self._buf.span(start=start)

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
        self: Image[dim,DType.float32],
        msg: String,
        sizes: Self.Vec[Int],
        head: InlineArray[Float32, samples],
        tail: InlineArray[Float32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5
    ):
        self._buf.assert_info(msg, sizes, head, tail, hash, verbose=verbose, eps=eps)

    # TEMP
    fn assert_data[err_fn: ErrFnFloat32 = err_rel](
        mut self: Image[dim,DType.float32],
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
            offset <= bound,
            "offset ", offset, " out of range [0,", bound, "]"
        )

    fn load[width: Int](self, offset: Int, out v: Self.PixelVec[width]):
        self._check_vector_offset[width](offset)
        v = self.span(start=offset)
            .unsafe_ptr()
            .load[width=width]()

    fn store[width: Int](mut self, offset: Int, v: Self.PixelVec[width]):
        self._check_vector_offset[width](offset)
        self.span(start=offset)
            .unsafe_ptr()
            .store[width=width](v)
        
    fn pixels_read[
        func: fn[width: Int](Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](ref self):

        @parameter
        fn loader[width: Int](offset: Int):
            var v = self.load[width](offset)
            func[width](v)

        vectorize[loader, width](self.num_pixels())

    fn pixels_read_write[
        func: fn[width: Int](mut p: Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](mut self):

        @parameter
        fn loader[width: Int](offset: Int):
            var v = self.load[width](offset)
            func[width](v)
            self.store[width](offset, v)

        vectorize[loader, width](self.num_pixels())

    fn fill(mut self, v: Self.PixelType):

        @always_inline
        @parameter
        fn func[width: Int](mut p: Self.PixelVec[width]):
            p = v
            
    fn copy(self, *, center: Self.Vec[Int], padding: Self.PixelType, mut to: Self):

        @parameter
        fn func(i: Self.Vec[Int]):
            var to_center = to.sizes()//2
            # NOTE: don't try to compute `to_center` outside the loop: it causes a compiler bug!!
            to[i] = self.get(i + center - to_center).or_else(padding)

        to.iterate[func]()

    # TODO: move these function to struct extension functions? when that gets released?

    fn mean(self: Image[dim,DType.float32]) -> Float64:
        """Returns mean of the entire image, in double precision."""
        return self.mean(mask = AllMask())

    fn mean[
        M: MaskReal, //
    ](
        self: Image[dim,DType.float32],
        *,
        mask: M
    ) -> Float64:
        """Returns mean of the masked region, in double precision."""

        # since we can't divide by zero
        if self.num_pixels() <= 0:
            return 0

        var sum: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        @parameter
        fn func(i: Vec[Int,dim]):
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                sum += Float64(self[i])

        self.iterate[func]()

        return sum/Float64(num_pixels_matched)

    fn mean_variance(self: Image[dim,DType.float32]) -> Tuple[Float32, Float32]:
        """Returns mean and variance of the entire image, in single precision."""
        return self.mean_variance(mask = AllMask())

    fn mean_variance[
        M: MaskReal, //
    ](
        self: Image[dim,DType.float32],
        *,
        mask: M
    ) -> Tuple[Float32, Float32]:
        """Returns mean and variance of the masked region, in single precision."""

        # since we can't divide by zero
        if self.num_pixels() <= 0:
            return (0, 0)

        var sum: Float64 = 0
        var sum_of_squares: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        @parameter
        fn func(i: Vec[Int,dim]):
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                # TODO: this is higher-precision, and possibly more efficient, but doesn't match the original csp
                # var p = Float64(self[i])
                # sum += p
                # p *= p
                # sum_of_squares += p
                # TEMP: for now, do the lower-precision thing, to match the original csp exactly
                var p = self[i]
                sum += Float64(p)
                sum_of_squares += Float64(p*p)

        self.iterate[func]()

        var n = Float64(num_pixels_matched)
        var mean = Float32(sum/n)
        # TODO: this is higher-precision, but doesn't match the original csp
        #var variance = abs(Float32( sum_of_squares/n - (sum/n)*(sum/n) ))
        # TEMP: for now, do the lower-precision thing, to match the original csp exactly
        var variance = abs(Float32( sum_of_squares/n - Float64(mean*mean) ))
        return (mean, variance)
