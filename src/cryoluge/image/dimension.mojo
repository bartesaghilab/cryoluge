
from os import abort
from sys.info import size_of
from complex import ComplexFloat32


@fieldwise_init
struct ImageDimension(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var rank: UInt

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


fn expect_at_least_rank[dim: ImageDimension, rank: UInt]():
    constrained[
        dim.rank >= rank,
        String("Expected dimension of at least rank ", rank, " but got ", dim)
    ]()


fn expect_num_arguments[dim: ImageDimension, count: UInt]():
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
    var _sizes: Self.VecD[UInt]
    var _strides: Self.VecD[UInt]
    """the element strides, not the byte strides"""
    var _buf: ByteBuffer

    alias VecD = VecD[_,dim]
    alias _elem_size = size_of[T]()

    fn __init__(out self, sizes: Self.VecD[UInt]):
        self._sizes = sizes.copy()
        @parameter
        if dim == ImageDimension.D1:
            var sx = self._sizes.x()
            self._strides = Self.VecD[UInt](x=1)
            self._buf = ByteBuffer(sx*Self._elem_size)
        elif dim == ImageDimension.D2:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            self._strides = Self.VecD[UInt](x=1, y=sx)
            self._buf = ByteBuffer(sx*sy*Self._elem_size)
        elif dim == ImageDimension.D3:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            var sz = self._sizes.z()
            self._strides = Self.VecD[UInt](x=1, y=sx, z=sx*sy)
            self._buf = ByteBuffer(sx*sy*sz*Self._elem_size)
        else:
            return unrecognized_dimension[dim,Self]()

    fn rank(self) -> UInt:
        return dim.rank

    fn num_elements(self) -> UInt:
        var count: UInt = 1
        @parameter
        for d in range(dim.rank):
            count *= self._sizes[d]
        return count

    fn sizes(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._sizes)]] Self.VecD[UInt]:
        return self._sizes

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf)]]:
        return self._buf.span()

    fn _start(self) -> UnsafePointer[T]:
        return self._buf._p.bitcast[T]()

    fn unsafe_ptr(self, *, offset: Int = 0) -> UnsafePointer[T]:
        debug_assert(
            offset >= 0 and offset < self.num_elements(),
            String("Offset ", offset, " out of range [0,", self.num_elements(), ")")
        )
        return self._start() + offset

    fn _offset(self, i: Self.VecD[UInt], out offset: UInt):
        
        alias d_names = InlineArray[String, 3]("x", "y", "z")

        offset = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(coord < size, d_names[d], "=", coord, " out of range [0,", size, ")")
            offset += coord*self._strides[d]

    fn _maybe_offset(self, i: Self.VecD[Int]) -> Optional[UInt]:

        var offset: UInt = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            if coord < Int(size):
                offset += coord*self._strides[d]
            else:
                return None
        
        return offset

    fn _check_offset(self, i: UInt):
        debug_assert(i < self.num_elements(), "i=", i, " out of range [0,", self.num_elements(), ")")

    fn __getitem__(self, *, i: UInt, out v: T):
        self._check_offset(i)
        v = (self._start() + i)[].copy()

    fn __getitem__(self, i: Self.VecD[UInt], out v: T):
        v = self[i=self._offset(i)]

    fn __getitem__(self, *, x: UInt, out v: T):
        expect_at_least_rank[dim, 1]()
        v = self[Self.VecD[UInt](x=x)]

    fn __getitem__(self, *, x: UInt, y: UInt, out v: T):
        expect_at_least_rank[dim, 2]()
        v = self[Self.VecD[UInt](x=x, y=y)]

    fn __getitem__(self, *, x: UInt, y: UInt, z: UInt, out v: T):
        expect_at_least_rank[dim, 3]()
        v = self[Self.VecD[UInt](x=x, y=y, z=z)]

    fn __setitem__(mut self, *, i: UInt, v: T):
        self._check_offset(i)
        (self._start() + i)[] = v.copy()

    fn __setitem__(mut self, i: Self.VecD[UInt], v: T):
        self[i=self._offset(i)] = v

    fn __setitem__(mut self, *, x: UInt, v: T):
        expect_at_least_rank[dim, 1]()
        self[Self.VecD[UInt](x=x)] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, v: T):
        expect_at_least_rank[dim, 2]()
        self[Self.VecD[UInt](x=x, y=y)] = v

    fn __setitem__(mut self, *, x: UInt, y: UInt, z: UInt, v: T):
        expect_at_least_rank[dim, 3]()
        self[Self.VecD[UInt](x=x, y=y, z=z)] = v

    fn get(self, i: Self.VecD[Int]) -> Optional[T]:
        var offset = self._maybe_offset(i)
        if offset is None:
            return None
        return self[i=offset.value()]

    fn iterate[
        func: fn (i: Self.VecD[UInt]) capturing
    ](self):

        @parameter
        if dim == ImageDimension.D1:
            
            for x in range(self.sizes().x()):
                func(Self.VecD[UInt](x=x))

        elif dim == ImageDimension.D2:
            
            for y in range(self.sizes().y()):
                for x in range(self.sizes().x()):
                    func(Self.VecD[UInt](x=x, y=y))

        elif dim == ImageDimension.D3:

            for z in range(self.sizes().z()):
                for y in range(self.sizes().y()):
                    for x in range(self.sizes().x()):
                        func(Self.VecD[UInt](x=x, y=y, z=z))

        else:
            unrecognized_dimension[dim]()

    # TEMP
    fn assert_info[samples: Int](
        self: DimensionalBuffer[dim,Float32],
        msg: String,
        sizes: Self.VecD[UInt],
        head: InlineArray[Float32, samples],
        tail: InlineArray[Float32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False
    ):

        # check the sizes
        debug_assert(
            self.sizes() == sizes,
            "expected sizes=", sizes, ", but got sizes=", self.sizes()
        )

        # check the samples
        debug_assert(
            self.num_elements() >= 3,
            msg, ": buffer too small for ", samples, " samples"
        )
        @parameter
        for s in range(samples):
            assert_sample(String(msg, ": head[", s, "]"), self[i=s], head[s])
        @parameter
        for s in range(samples):
            s2 = self.num_elements() - samples + s
            assert_sample(String(msg, ": tail[", s, "]"), self[i=s2], tail[s])

        # check the hash
        var obs_hash: UInt64 = 0
        for i in range(self.num_elements()):
            obs_hash *= 37
            obs_hash += UInt64(UnsafePointer(to=self[i=i]).bitcast[UInt32]()[])
        debug_assert(
            obs_hash == hash,
            "Hash mismatch: obs=", obs_hash, ", exp=", hash
        )

        if verbose:
            print(String("info OK: ", msg, ": sizes=", sizes, ", hash=", hash))

    # TEMP
    fn assert_info[samples: Int](
        self: DimensionalBuffer[dim,ComplexFloat32],
        msg: String,
        sizes: Self.VecD[UInt],
        head: InlineArray[ComplexFloat32, samples],
        tail: InlineArray[ComplexFloat32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False
    ):

        # check the sizes
        debug_assert(
            self.sizes() == sizes,
            "expected sizes=", sizes, ", but got sizes=", self.sizes()
        )

        # check the samples
        debug_assert(
            self.num_elements() >= 3,
            msg, ": buffer too small for ", samples, " samples"
        )
        @parameter
        for s in range(samples):
            assert_sample(String(msg, ": head[", s, "]"), self[i=s], head[s])
        @parameter
        for s in range(samples):
            s2 = self.num_elements() - samples + s
            assert_sample(String(msg, ": tail[", s, "]"), self[i=s2], tail[s])

        # check the hash
        var obs_hash: UInt64 = 0
        for i in range(self.num_elements()*2):
            var p = UnsafePointer(to=self[i=i]).bitcast[UInt32]()
            obs_hash *= 37
            obs_hash += UInt64(p[])
            p += 1
            obs_hash *= 37
            obs_hash += UInt64(p[])
        debug_assert(
            obs_hash == hash,
            "Hash mismatch: obs=", obs_hash, ", exp=", hash
        )

        if verbose:
            print(String("info OK: ", msg, ": sizes=", sizes, ", hash=", hash))

# TEMP
fn assert_sample(
    msg: String,
    obs: Float32,
    exp: Float32,
    eps: Float32 = 1e-5   
):
    debug_assert(
        abs(obs - exp) <= eps,
        msg, ": obs=", obs, "  exp=", exp
    )

# TEMP
fn assert_sample(
    msg: String,
    obs: ComplexFloat32,
    exp: ComplexFloat32,
    eps: Float32 = 1e-5   
):
    debug_assert(
        abs(obs.re - exp.re) <= eps and abs(obs.im - exp.im) <= eps,
        msg, ": obs=", obs, "  exp=", exp
    )
