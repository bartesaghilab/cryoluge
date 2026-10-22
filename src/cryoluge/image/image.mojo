
from std.sys import simd_width_of
from std.algorithm import vectorize
from std.math import sqrt

from cryoluge.math import Vec
from cryoluge.math.dimension import DimensionalIterator
from cryoluge.image.dimension import DimensionalBuffer
from cryoluge.image.mask import MaskReal, MaskFourier, AllMask


struct Image[
    dim: Int,
    dtype: DType
](Copyable, Movable)
    where dim > 0:
    
    var _buf: DimensionalBuffer[Self.dim,Self.PixelType]

    comptime Vec = Vec[Self.dim,_]
    comptime PixelType = Scalar[Self.dtype]
    comptime PixelVec = SIMD[Self.dtype,_]
    comptime pixel_vec_max_slen = simd_width_of[Self.dtype]()

    def __init__(
        out self,
        sizes: Self.Vec[.int],
        *,
        uninitialized: Bool,
        alignment: Optional[Int] = None
    ):
        self._buf = DimensionalBuffer[Self.dim,Self.PixelType](sizes, alignment=alignment)

    def __init__(
        out self,
        sizes: Self.Vec[.int],
        *,
        fill: Self.PixelType,
        alignment: Optional[Int] = None
    ):
        self = Self(sizes, uninitialized=True, alignment=alignment)
        self.fill(fill)
    
    @inline(.always)
    def dimension(self) -> Int:
        return self._buf.dimension()

    @inline(.always)
    def num_pixels(self) -> Int:
        return self._buf.num_elements()

    @inline(.always)
    def sizes(self) -> ref [origin_of(self._buf._sizes)] Self.Vec[.int]:
        return self._buf.sizes()

    @inline(.always)
    def span_bytes(
        ref self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte, origin_of(self._buf._buf._alloc._alloc)]:
        return self._buf.span_bytes(start=start, length=length)

    @inline(.always)
    def span(
        ref self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Self.PixelType, origin_of(self._buf._buf._alloc._alloc)]:
        return self._buf.span(start=start, length=length)

    @inline(.always)
    def alignment(self) -> Int:
        return self._buf.alignment()

    @inline(.always)
    def __getitem__(self, *, i: Int, out v: Self.PixelType):
        v = self._buf[i=i]

    @inline(.always)
    def __getitem__(self, i: Self.Vec[.int], out v: Self.PixelType):
        v = self._buf[i]

    @inline(.always)
    def __setitem__(mut self, *, i: Int, v: Self.PixelType):
        self._buf[i=i] = v

    @inline(.always)
    def __setitem__(mut self, i: Self.Vec[.int], v: Self.PixelType):
        self._buf[i] = v

    @inline(.always)
    def get(self, i: Self.Vec[.int]) -> Optional[Self.PixelType]:
        return self._buf.get(i)

    def pixels(self) -> ImageIterator[Self.dim, Self.dtype]:
        return ImageIterator[Self.dim,Self.dtype](self._buf, self.num_pixels())

    @inline(.always)
    def offset(self, i: Self.Vec[.int]) -> Int:
        return self._buf.offset(i)

    @inline(.always)
    def maybe_offset(self, i: Self.Vec[.int]) -> Optional[Int]:
        return self._buf.maybe_offset(i)

    @inline(.always)
    def load[slen: SIMDLength](self, offset: Int, out v: Self.PixelVec[slen]):
        v = self.span(start=offset, length=Int(slen))
            .unsafe_ptr()
            .unsafe_load[width=slen]()

    @inline(.always)
    def load[slen: SIMDLength](self, i: Self.Vec[.int], out v: Self.PixelVec[slen]):
        v = self.load[slen](self.offset(i))

    @inline(.always)
    def store[slen: SIMDLength](mut self, offset: Int, v: Self.PixelVec[slen]):
        self.span(start=offset, length=Int(slen))
            .unsafe_ptr()
            .unsafe_store[width=slen](v)

    @inline(.always)
    def store[slen: SIMDLength](mut self, i: Self.Vec[.int], v: Self.PixelVec[slen]):
        self.store[slen](self.offset(i), v)

    def fill(mut self, v: Self.PixelType):

        @inline(.always)
        def func[slen: SIMDLength](offset: Int) {mut self, v}:
            self.store[slen](offset, v)

        self.pixels().vectorize(func)
    
    def copy(
        self,
        *,
        center: Self.Vec[.int],
        padding: Self.PixelType,
        mut to: Self
    ):

        var to_center = to.sizes()//2

        @inline(.always)
        def func(i: Vec[Self.dim,.int]) {self, center, padding, mut to, to_center}:
            to[i] = self.get(i + center - to_center).or_else(padding)

        to.pixels().iterate(func)
        # TODO: vectorize?

    def mean(self) -> Scalar[Self.dtype]:
        """Returns the mean of the entire image."""
        return self.mean(mask = AllMask())

    def mean[M: MaskReal, //](
        self,
        *,
        mask: M
    ) -> Scalar[Self.dtype]:
        """Returns the mean of the masked region."""

        var sum: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        def func(i: Self.Vec[.int]) {self, mask, mut num_pixels_matched, mut sum}:
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                sum += Float64(self[i])

        self.pixels().iterate(func)

        # since we can't divide by zero
        if num_pixels_matched <= 0:
            return 0

        return Scalar[Self.dtype](sum/Float64(num_pixels_matched))

    def mean_variance(self) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]:
        """Returns the mean and variance of the entire image."""
        return self.mean_variance(mask = AllMask())

    def mean_variance[M: MaskReal, //](
        self,
        *,
        mask: M
    ) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]:
        """Returns the mean and variance of the masked region."""

        var sum: Float64 = 0
        var sum_of_squares: Float64 = 0
        var num_pixels_matched: Int = 0

        # TODO: any chance we can vectorize this?

        def func(i: Self.Vec[.int]) {self, mask, mut num_pixels_matched, mut sum, mut sum_of_squares}:
            if mask.includes(i, self.sizes()):
                num_pixels_matched += 1
                var p = Float64(self[i])
                sum += p
                p *= p
                sum_of_squares += p

        self.pixels().iterate(func)

        # since we can't divide by zero
        if num_pixels_matched <= 0:
            return (0, 0)

        var n = Float64(num_pixels_matched)
        var mean = Scalar[Self.dtype](sum/n)
        var variance = abs(Scalar[Self.dtype]( sum_of_squares/n - (sum/n)**2 ))
        return (mean, variance)

    def range(self) -> Tuple[Scalar[Self.dtype],Scalar[Self.dtype]]:
        return self.range(mask=AllMask())

    def range[M: MaskReal, //](
        self,
        *,
        mask: M
    ) -> Tuple[Scalar[Self.dtype],Scalar[Self.dtype]]:
        
        var min = Scalar[Self.dtype].MAX
        var max = Scalar[Self.dtype].MIN

        # TODO: vectorize?
        def func(i: Self.Vec[.int]) {self, mask, mut min, mut max}:
            if mask.includes(i, self.sizes()):
                var p = self[i]
                if p < min:
                    min = p
                elif p > max:
                    max = p
        
        self.pixels().iterate(func)

        return (min, max)

    def replace_outliers(
        mut self,
        *,
        mean: Scalar[Self.dtype],
        variance: Scalar[Self.dtype],
        gt_sigmas: Scalar[Self.dtype],
        replace_with: Scalar[Self.dtype]
    ):
        """Replaces pixels with sigma (relative to `mean` and `variance`) greater than `gt_sigmas` with `replace_with`."""

        var stddev = sqrt(variance)
        var max = mean + gt_sigmas*stddev
        var min = mean - gt_sigmas*stddev

        def func(i: Self.Vec[.int]) {mut self, min, max, replace_with}:
            var p = self[i]
            if p > max or p < min:
                self[i] = replace_with

        self.pixels().iterate(func)


struct ImageIterator[dim: Int, dtype: DType](
    Movable
) where dim > 0:

    var _iter: DimensionalIterator[Self.dim]
    var _num_pixels: Int

    comptime max_slen = simd_width_of[Self.dtype]()

    def __init__(out self, buf: DimensionalBuffer[Self.dim,_], num_pixels: Int):
        self._iter = buf.elements()
        self._num_pixels = num_pixels

    def iterate(
        self,
        func: Some[def (i: Vec[Self.dim,.int])]
    ):
        self._iter.iterate(func)
    
    def vectorize[slen: SIMDLength = Self.max_slen](
        self,
        func: Some[def[slen: SIMDLength](index: Int)]
    ):
        @inline(.always)
        def func2[slen: Int](offset: Int) {func}:
            func[slen](offset)

        vectorize[slen](self._num_pixels, func2)

    def vectorize[slen: SIMDLength = Self.max_slen](
        self,
        func: Some[def[slen: SIMDLength](i: Vec[Self.dim,.int])]
    ) where Self.dim > 0:
        """
        Iterates over the outer dimensions (eg, yz) and yields pixel vectors over the first (eg, x) dimension.
        """

        @inline(.always)
        def outer_dims(outer_indices: Vec[Self.dim - 1, .int]) {self, func}:

            @inline(.always)
            def inner_dim[slen: Int](offset: Int) {func, outer_indices}:
                var indices = outer_indices.lift_front[Self.dim](Vec[1](x=offset))
                func[slen](indices)

            # then vectorize the first dimension, eg, x
            vectorize[slen](self._iter._sizes[0], inner_dim)

        # iterate over the outer dimensions first, eg, yz
        DimensionalIterator(self._iter._sizes.project_last[Self.dim - 1]())
            .iterate(outer_dims)
