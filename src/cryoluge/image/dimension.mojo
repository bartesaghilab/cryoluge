
from os import abort
from sys.info import size_of
from complex import ComplexFloat32
from memory import bitcast


@fieldwise_init
struct ImageDimension(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var rank: Int

    # NOTE: mojo's grammar doesn't allow `1D`, `2D`, etc identifiers
    alias D1 = Self(1)
    alias D2 = Self(2)
    alias D3 = Self(3)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.rank == rhs.rank

    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.D1:
            writer.write("D1")
        elif self == Self.D2:
            writer.write("D2")
        elif self == Self.D3:
            writer.write("D3")
        else:
            writer.write("Unknown(", self.rank, ")")

    fn __str__(self) -> String:
        return String.write(self)


fn unrecognized_dimension[dim: ImageDimension, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Unrecognized dimensionality: ", dim)]()
    return abort[T]()


fn unimplemented_dimension[dim: ImageDimension, T: AnyType = NoneType._mlir_type]() -> T:
    constrained[False, String("Dimension not implemented yet: ", dim)]()
    return abort[T]()


fn expect_at_least_rank[dim: ImageDimension, rank: Int]():
    constrained[
        dim.rank >= rank,
        String("Expected dimension of at least rank ", rank, " but got ", dim)
    ]()


fn expect_num_arguments[dim: ImageDimension, count: Int]():
    constrained[
        dim.rank == count,
        String(dim, " expects ", dim.rank, " argument(s), but got ", count, " instead")
    ]()


struct DimensionalBuffer[
    dim: ImageDimension,
    T: Copyable & Movable
](
    Copyable,
    Movable
):
    var _sizes: Self.VecD[Int]
    var _strides: Self.VecD[Int]
    """the element strides, not the byte strides"""
    var _buf: ByteBuffer

    alias VecD = VecD[_,dim]
    alias _elem_size = size_of[T]()

    fn __init__(out self, sizes: Self.VecD[Int], *, alignment: Optional[Int] = None):
        self._sizes = sizes.copy()
        @parameter
        if dim == ImageDimension.D1:
            var sx = self._sizes.x()
            self._strides = Self.VecD[Int](x=1)
            self._buf = ByteBuffer(sx*Self._elem_size, alignment=alignment)
        elif dim == ImageDimension.D2:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            self._strides = Self.VecD[Int](x=1, y=sx)
            self._buf = ByteBuffer(sx*sy*Self._elem_size, alignment=alignment)
        elif dim == ImageDimension.D3:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            var sz = self._sizes.z()
            self._strides = Self.VecD[Int](x=1, y=sx, z=sx*sy)
            self._buf = ByteBuffer(sx*sy*sz*Self._elem_size, alignment=alignment)
        else:
            return unrecognized_dimension[dim,Self]()

    fn rank(self) -> Int:
        return dim.rank

    fn num_elements(self) -> Int:
        var count: Int = 1
        @parameter
        for d in range(dim.rank):
            count *= self._sizes[d]
        return count

    fn sizes(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._sizes)]] Self.VecD[Int]:
        return self._sizes

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf)]]:
        return self._buf.span()

    fn alignment(self) -> Int:
        return self._buf.alignment()

    fn _start(self) -> UnsafePointer[T]:
        return self._buf._p.bitcast[T]()

    fn unsafe_ptr(self, *, offset: Int = 0) -> UnsafePointer[T]:
        debug_assert(
            offset >= 0 and offset < self.num_elements(),
            String("Offset ", offset, " out of range [0,", self.num_elements(), ")")
        )
        return self._start() + offset

    fn _offset(self, i: Self.VecD[Int], out offset: Int):
        
        alias d_names = InlineArray[String, 3]("x", "y", "z")

        offset = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(coord >= 0 and coord < size, d_names[d], "=", coord, " out of range [0,", size, ")")
            offset += coord*self._strides[d]

    fn _maybe_offset(self, i: Self.VecD[Int]) -> Optional[Int]:

        var offset: Int = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            if coord < size:
                offset += coord*self._strides[d]
            else:
                return None
        
        return offset

    fn _check_offset(self, i: Int):
        debug_assert(i >= 0 and i < self.num_elements(), "i=", i, " out of range [0,", self.num_elements(), ")")

    fn __getitem__(self, *, i: Int, out v: T):
        self._check_offset(i)
        v = (self._start() + i)[].copy()

    fn __getitem__(self, i: Self.VecD[Int], out v: T):
        v = self[i=self._offset(i)]

    fn __getitem__(self, *, x: Int, out v: T):
        expect_at_least_rank[dim, 1]()
        v = self[Self.VecD(x=x)]

    fn __getitem__(self, *, x: Int, y: Int, out v: T):
        expect_at_least_rank[dim, 2]()
        v = self[Self.VecD(x=x, y=y)]

    fn __getitem__(self, *, x: Int, y: Int, z: Int, out v: T):
        expect_at_least_rank[dim, 3]()
        v = self[Self.VecD[Int](x=x, y=y, z=z)]

    fn __setitem__(mut self, *, i: Int, v: T):
        self._check_offset(i)
        (self._start() + i)[] = v.copy()

    fn __setitem__(mut self, i: Self.VecD[Int], v: T):
        self[i=self._offset(i)] = v

    fn __setitem__(mut self, *, x: Int, v: T):
        expect_at_least_rank[dim, 1]()
        self[Self.VecD[Int](x=x)] = v

    fn __setitem__(mut self, *, x: Int, y: Int, v: T):
        expect_at_least_rank[dim, 2]()
        self[Self.VecD[Int](x=x, y=y)] = v

    fn __setitem__(mut self, *, x: Int, y: Int, z: Int, v: T):
        expect_at_least_rank[dim, 3]()
        self[Self.VecD[Int](x=x, y=y, z=z)] = v

    fn get(self, i: Self.VecD[Int]) -> Optional[T]:
        var offset = self._maybe_offset(i)
        if offset is None:
            return None
        return self[i=offset.value()]

    fn iterate[
        func: fn (i: Self.VecD[Int]) capturing
    ](self):

        @parameter
        if dim == ImageDimension.D1:
            
            for x in range(self.sizes().x()):
                func(Self.VecD[Int](x=x))

        elif dim == ImageDimension.D2:
            
            for y in range(self.sizes().y()):
                for x in range(self.sizes().x()):
                    func(Self.VecD[Int](x=x, y=y))

        elif dim == ImageDimension.D3:

            for z in range(self.sizes().z()):
                for y in range(self.sizes().y()):
                    for x in range(self.sizes().x()):
                        func(Self.VecD[Int](x=x, y=y, z=z))

        else:
            unrecognized_dimension[dim]()

    # TEMP
    fn assert_info[samples: Int](
        self: DimensionalBuffer[dim,Float32],
        msg: String,
        sizes: Self.VecD[Int],
        head: InlineArray[Float32, samples],
        tail: InlineArray[Float32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False
    ):

        # check the sizes
        debug_assert(
            self.sizes() == sizes,
            msg, ": expected sizes=", sizes, ", but got sizes=", self.sizes()
        )

        # check the samples
        var samples_match = True
        @parameter
        for s in range(samples):
            samples_match = samples_match and sample_eq(self[i=s], head[s])
        @parameter
        for s in range(samples):
            s2 = self.num_elements() - samples + s
            samples_match = samples_match and sample_eq(self[i=s2], tail[s])
        debug_assert(samples_match, msg, ": ", self.dump_samples(head, tail))

        # check the hash
        var obs_hash: UInt64 = 0
        for i in range(self.num_elements()):
            obs_hash *= 37
            obs_hash += UInt64(bitcast[DType.uint32](self[i=i]))
        debug_assert(
            obs_hash == hash,
            msg, ": Hash mismatch: obs=", obs_hash, ", exp=", hash
        )

        if verbose:
            print(String("info OK: ", msg, ": sizes=", sizes, ", hash=", hash))

    # TEMP
    fn assert_info[samples: Int](
        self: DimensionalBuffer[dim,ComplexFloat32],
        msg: String,
        sizes: Self.VecD[Int],
        head: InlineArray[ComplexFloat32, samples],
        tail: InlineArray[ComplexFloat32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False
    ):

        # check the sizes
        debug_assert(
            self.sizes() == sizes,
            msg, ": expected sizes=", sizes, ", but got sizes=", self.sizes()
        )

        # check the samples
        debug_assert(
            self.num_elements() >= 3,
            msg, ": buffer too small for ", samples, " samples"
        )
        var samples_match = True
        @parameter
        for s in range(samples):
            samples_match = samples_match and sample_eq(self[i=s], head[s])
        @parameter
        for s in range(samples):
            s2 = self.num_elements() - samples + s
            samples_match = samples_match and sample_eq(self[i=s2], tail[s])
        debug_assert(samples_match, msg, ": ", self.dump_samples(head, tail))

        # check the hash
        var obs_hash: UInt64 = 0
        for i in range(self.num_elements()):
            var v = self[i=i]
            obs_hash *= 37
            obs_hash += UInt64(bitcast[DType.uint32](v.re))
            obs_hash *= 37
            obs_hash += UInt64(bitcast[DType.uint32](v.im))
        debug_assert(
            obs_hash == hash,
            msg, ": Hash mismatch: obs=", obs_hash, ", exp=", hash
        )

        if verbose:
            print(String("info OK: ", msg, ": sizes=", sizes, ", hash=", hash))


    fn dump_samples[samples: Int, T: Stringable & Copyable & Movable](
        self: DimensionalBuffer[dim,T],
        head: InlineArray[T,samples],
        tail: InlineArray[T,samples]
    ) -> String:
        var msg = "Samples don't match!:\n"

        msg += "\thead obs = [ "
        @parameter
        for s in range(samples):
            @parameter
            if s > 0:
                msg += ",  "
            msg += String(self[i=s])
        msg += " ]\n"

        msg += "\thead exp = [ "
        @parameter
        for s in range(samples):
            @parameter
            if s > 0:
                msg += ",  "
            msg += String(head[s])
        msg += " ]\n"

        msg += "\ttail obs = [ "
        @parameter
        for s in range(samples):
            @parameter
            if s > 0:
                msg += ",  "
            s2 = self.num_elements() - samples + s
            msg += String(self[i=s2])
        msg += " ]\n"

        msg += "\ttail exp = [ "
        @parameter
        for s in range(samples):
            @parameter
            if s > 0:
                msg += ",   "
            msg += String(tail[s])
        msg += " ]"

        return msg


fn err_rel(obs: Float32, exp: Float32) -> Float32:
    if exp == 0:
        return abs(obs - exp)
    else:
        return abs(obs - exp)/abs(exp)

# TEMP
fn sample_eq(obs: Float32, exp: Float32, eps: Float32 = 1e-5   
) -> Bool:
    return err_rel(obs, exp) <= eps

# TEMP
fn sample_eq(obs: ComplexFloat32, exp: ComplexFloat32, eps: Float32 = 1e-5) -> Bool:
    return err_rel(obs.re, exp.re) <= eps and err_rel(obs.im, exp.im) <= eps
