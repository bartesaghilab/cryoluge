
from cryoluge.math import Vec
from cryoluge.math.error import ErrFn, err_rel
from cryoluge.io import ByteBuffer


struct Image[
    dim: Int,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    comptime Vec = Vec[dim,_]
    comptime PixelType = Scalar[dtype]
    comptime PixelVec = SIMD[dtype,_]
    comptime pixel_vec_max_width = simd_width_of[dtype]()

    def __init__(out self, sizes: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes, alignment=alignment)

    def __init__(out self, *, sx: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx), alignment=alignment)

    def __init__(out self, *, sx: Int, sy: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx, y=sy), alignment=alignment)

    def __init__(out self, *, sx: Int, sy: Int, sz: Int, alignment: Optional[Int] = None):
        self = Self(Self.Vec(x=sx, y=sy, z=sz), alignment=alignment)

    def rank(self) -> Int:
        return self._buf.rank()

    def num_pixels(self) -> Int:
        return self._buf.num_elements()

    def sizes(self) -> ref [origin_of(self._buf._sizes)] Self.Vec[Int]:
        return self._buf.sizes()

    def span_bytes(ref self) -> Span[Byte, origin_of(self._buf._buf)]:
        return self._buf.span_bytes()

    def span(ref self, *, start: Int = 0) -> Span[Self.PixelType, origin_of(self._buf._buf)]:
        return self._buf.span(start=start)

    def alignment(self) -> Int:
        return self._buf.alignment()

    def __getitem__(self, *, i: Int, out v: Self.PixelType):
        v = self._buf[i=i]

    def __getitem__(self, i: Self.Vec[Int], out v: Self.PixelType):
        v = self._buf[i]

    def __getitem__(self, *, x: Int, out v: Self.PixelType):
        v = self._buf[x=x]

    def __getitem__(self, *, x: Int, y: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y]

    def __getitem__(self, *, x: Int, y: Int, z: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y, z=z]

    def __setitem__(mut self, *, i: Int, v: Self.PixelType):
        self._buf[i=i] = v

    def __setitem__(mut self, i: Self.Vec[Int], v: Self.PixelType):
        self._buf[i] = v

    def __setitem__(mut self, *, x: Int, v: Self.PixelType):
        self._buf[x=x] = v

    def __setitem__(mut self, *, x: Int, y: Int, v: Self.PixelType):
        self._buf[x=x, y=y] = v

    def __setitem__(mut self, *, x: Int, y: Int, z: Int, v: Self.PixelType):
        self._buf[x=x, y=y, z=z] = v

    def get(self, i: Self.Vec[Int]) -> Optional[Self.PixelType]:
        return self._buf.get(i)

    def iterate[
        func: def (i: Self.Vec[Int]) capturing
    ](self):
        self._buf.iterate[func]()

    # TEMP: for comparing to images in other programs
    def assert_info[samples: Int](
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
    def assert_data[err_fn: ErrFn[dtype] = err_rel[dtype]](
        mut self,
        msg: String,
        path: String,
        *,
        verbose: Bool = False,
        eps: Scalar[dtype] = 1e-5,
        overwrite: Bool = False
    ) raises:
        self._buf.assert_data[dtype,err_fn](msg, path, verbose=verbose, eps=eps, overwrite=overwrite)

    def _check_vector_offset[width: Int](self, offset: Int):
        var bound = self.num_pixels() - width
        debug_assert(
            offset <= bound,
            "offset ", offset, " out of range [0,", bound, "]"
        )

    def load[width: Int](self, offset: Int, out v: Self.PixelVec[width]):
        self._check_vector_offset[width](offset)
        v = self.span(start=offset)
            .unsafe_ptr()
            .load[width=width]()

    # TEMP
    def load[width: Int](self, *, i: Vec[dim,Int], out v: Self.PixelVec[width]):
        var offset = self._buf._offset(i)
        v = self.span(start=offset)
            .unsafe_ptr()
            .load[width=width]()

    def store[width: Int](mut self, offset: Int, v: Self.PixelVec[width]):
        self._check_vector_offset[width](offset)
        self.span(start=offset)
            .unsafe_ptr()
            .store[width=width](v)
        
    def pixels_read[
        func: def[width: Int](Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](ref self):

        @parameter
        def loader[width: Int](offset: Int):
            var v = self.load[width](offset)
            func[width](v)

        vectorize[loader, width](self.num_pixels())

    def pixels_write[
        func: def[width: Int](out p: Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](mut self):

        @parameter
        def loader[width: Int](offset: Int):
            var v = func[width]()
            self.store[width](offset, v)

        vectorize[loader, width](self.num_pixels())

    def pixels_read_write[
        func: def[width: Int](mut p: Self.PixelVec[width]) capturing,
        width: Int = Self.pixel_vec_max_width
    ](mut self):

        @parameter
        def loader[width: Int](offset: Int):
            var v = self.load[width](offset)
            func[width](v)
            self.store[width](offset, v)

        vectorize[loader, width](self.num_pixels())

    def fill(mut self, v: Self.PixelType):

        @inline(.always)
        @parameter
        def func[width: Int](out p: Self.PixelVec[width]):
            p = v

        self.pixels_write[func]()
    
    def copy(self, *, center: Self.Vec[Int], padding: Self.PixelType, mut to: Self):

        @parameter
        def func(i: Self.Vec[Int]):
            var to_center = to.sizes()//2
            # NOTE: don't try to compute `to_center` outside the loop: it causes a compiler bug!!
            to[i] = self.get(i + center - to_center).or_else(padding)

        to.iterate[func]()

    # TODO: move these function to struct extension functions? when that gets released?

    def mean(self) -> Scalar[dtype]:
        """Returns the mean of the entire image."""
        return self.mean(mask = AllMask())

    def mean[
        M: MaskReal, //
    ](
        self,
        *,
        mask: M
    ) -> Scalar[dtype]:
        """Returns the mean of the masked region."""

        var sum: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        @parameter
        def func(i: Vec[dim,Int]):
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                sum += Float64(self[i])

        self.iterate[func]()

        # since we can't divide by zero
        if num_pixels_matched <= 0:
            return 0

        return Scalar[dtype](sum/Float64(num_pixels_matched))

    def mean_variance(self) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Returns the mean and variance of the entire image."""
        return self.mean_variance(mask = AllMask())

    def mean_variance[
        M: MaskReal, //
    ](
        self,
        *,
        mask: M
    ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Returns the mean and variance of the masked region."""

        var sum: Float64 = 0
        var sum_of_squares: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        @parameter
        def func(i: Vec[dim,Int]):
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                var p = Float64(self[i])
                sum += p
                p *= p
                sum_of_squares += p

        self.iterate[func]()

        # since we can't divide by zero
        if num_pixels_matched <= 0:
            return (0, 0)

        var n = Float64(num_pixels_matched)
        var mean = Scalar[dtype](sum/n)
        var variance = abs(Scalar[dtype]( sum_of_squares/n - (sum/n)**2 ))
        return (mean, variance)

    def range(self) -> Tuple[Scalar[dtype],Scalar[dtype]]:
        return self.range(mask=AllMask())

    def range[
        M: MaskReal, //
    ](
        self,
        *,
        mask: M
    ) -> Tuple[Scalar[dtype],Scalar[dtype]]:
        
        var min = Scalar[dtype].MAX
        var max = Scalar[dtype].MIN

        # TODO: vectorize?
        @parameter
        def func(i: Vec[dim,Int]):
            if mask.includes(i, self.sizes()):
                var p = self[i]
                if p < min:
                    min = p
                elif p > max:
                    max = p
        
        self.iterate[func]()

        return (min, max)

    def replace_outliers(
        mut self,
        *,
        mean: Scalar[dtype],
        variance: Scalar[dtype],
        gt_sigmas: Scalar[dtype],
        replace_with: Scalar[dtype]
    ):
        """Replaces pixels with sigma (relative to `mean` and `variance`) greater than `gt_sigmas` with `replace_with`."""

        var stddev = sqrt(variance)
        var max = mean + gt_sigmas*stddev
        var min = mean - gt_sigmas*stddev

        @parameter
        def func(i: Vec[dim,Int]):
            var p = self[i]
            if p > max or p < min:
                self[i] = replace_with

        self.iterate[func]()
            
        # TEMP: extend lifetimes of captures vars to work around compiler bug
        _ = min
        _ = max
