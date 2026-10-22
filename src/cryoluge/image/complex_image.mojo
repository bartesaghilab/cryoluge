
from std.complex import ComplexSIMD
from std.algorithm import vectorize
from std.sys import simd_width_of

from cryoluge.math import Vec, ComplexScalar, ladder
from cryoluge.math.dimension import DimensionalIterator
from cryoluge.image.image import ImageIterator
from cryoluge.image.dimension import DimensionalBuffer
from cryoluge.math import complex


struct ComplexImage[
    dim: Int,
    dtype: DType
](Copyable, Movable)
    where dim > 0:

    var _buf: DimensionalBuffer[Self.dim,Self.PixelType]

    comptime Vec = Vec[Self.dim,_]
    comptime PixelType = ComplexScalar[Self.dtype]
    comptime PixelVec = ComplexSIMD[Self.dtype,_]
    comptime ScalarType = Scalar[Self.dtype]
    comptime ScalarVec = SIMD[Self.dtype,_]
    comptime pixel_vec_max_slen = simd_width_of[Self.dtype]()

    def __init__(
        out self,
        sizes: Self.Vec[.int],
        *,
        uninitialized: Bool,
        alignment: Optional[Int] = None
    ):
        self._buf = DimensionalBuffer[Self.dim,Self.PixelType](sizes, alignment=alignment)
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
    ) -> Span[Byte,origin_of(self._buf._buf)]:
        return self._buf.span_bytes(start=start, length=length)

    @inline(.always)
    def span(
        ref self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Self.PixelType,origin_of(self._buf._buf)]:
        return self._buf.span(start=start, length=length)

    @inline(.always)
    def alignment(self) -> Optional[Int]:
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

    @inline(.always)
    def get[*, or_else: Self.PixelType](self, i: Self.Vec[.int], out v: Self.PixelType):
        v = self._buf.get[or_else=or_else](i)

    def pixels(self) -> ImageIterator[Self.dim, Self.dtype]:
        return ImageIterator[Self.dim,Self.dtype](self._buf, self.num_pixels())

    @inline(.always)
    def load[slen: SIMDLength](self, offset: Int, out v: Self.PixelVec[slen]):

        # load the interleaved data
        var interleaved = self.span(start=offset, length=Int(slen))
            .unsafe_ptr()
            .unsafe_bitcast[Self.ScalarType]()
            .unsafe_load[slen*2]()
            # NOTE: mojo's compiler is smart enough to handle vector operations
            #       on sizes larger than the machine limits, as long as it's a power of 2

        # de-interleave
        var (real, imag) = interleaved.deinterleave()

        # build the complex vector
        v = Self.PixelVec[slen](
            re = rebind[Self.ScalarVec[slen]](real),
            im = rebind[Self.ScalarVec[slen]](imag)
        )

    @inline(.always)
    def load[slen: SIMDLength](self, *, i: Self.Vec[.int], out v: Self.PixelVec[slen]):
        v = self.load[slen](self._buf.offset(i))

    @inline(.always)
    def store[slen: SIMDLength](mut self, offset: Int, v: Self.PixelVec[slen]):

        # interleave
        var interleaved = rebind[Self.ScalarVec[slen*2]]( v.re.interleave(v.im) )

        # store the interleaved data
        self.span(start=offset, length=Int(slen))
            .unsafe_ptr()
            .unsafe_bitcast[Self.ScalarType]()
            .unsafe_store[slen*2](interleaved)

    def fill(mut self, v: Self.PixelType):

        @inline(.always)
        def func[slen: SIMDLength](offset: Int) {mut self, v}:
            self.store[slen](offset, complex.splat[slen](v))

        self.pixels().vectorize(func)
    
    def multiply[slen: SIMDLength = Self.pixel_vec_max_slen](mut self, factor: Self.ScalarType):

        def func[slen: SIMDLength](offset: Int) {mut self, factor}:
            var p = self.load[slen](offset)
            p *= factor
            self.store[slen](offset, p)

        self.pixels().vectorize[slen](func)
