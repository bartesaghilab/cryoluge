
from os import abort
from sys.info import size_of
from complex import ComplexFloat32
from memory import bitcast, memcpy

from cryoluge.io import FileReader, Endian


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
        return self._sizes.product()

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
            "Offset ", offset, " out of range [0,", self.num_elements(), ")"
        )
        return self._start() + offset

    fn _offset(self, i: Self.VecD[Int], out offset: Int):
        
        alias d_names = InlineArray[String, 3]("x", "y", "z")

        offset = 0

        @parameter
        for d in range(dim.rank):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(
                coord >= 0 and coord < size,
                d_names[d], "=", coord, " out of range [0,", size, ")"
            )
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
        debug_assert(
            i >= 0 and i < self.num_elements(),
            "i=", i, " out of range [0,", self.num_elements(), ")"
        )

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
    fn assert_info[samples: Int, err_fn: ErrFnFloat32 = err_rel](
        self: DimensionalBuffer[dim,Float32],
        msg: String,
        sizes: Self.VecD[Int],
        head: InlineArray[Float32, samples],
        tail: InlineArray[Float32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5
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
        var obs_head = self._sample[samples](0)
        var obs_tail = self._sample[samples](self.num_elements() - samples)
        var err_head = _err[err_fn](obs_head, head)
        var err_tail = _err[err_fn](obs_tail, tail)
        debug_assert(
            _err_small(err_head, eps) and _err_small(err_tail, eps),
            msg, ": Samples don't match!:",
            "\n\thead obs = ", self._render_samples(obs_head),
            "\n\thead exp = ", self._render_samples(head),
            "\n\thead err = ", self._render_samples(err_head),
            "\n\ttail obs = ", self._render_samples(obs_tail),
            "\n\ttail exp = ", self._render_samples(tail),
            "\n\ttail err = ", self._render_samples(err_tail),
        )

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
    fn assert_info[samples: Int, err_fn: ErrFnFloat32 = err_rel](
        self: DimensionalBuffer[dim,ComplexFloat32],
        msg: String,
        sizes: Self.VecD[Int],
        head: InlineArray[ComplexFloat32, samples],
        tail: InlineArray[ComplexFloat32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5
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
        var obs_head = self._sample[samples](0)
        var obs_tail = self._sample[samples](self.num_elements() - samples)
        var err_head = _err[err_fn](obs_head, head)
        var err_tail = _err[err_fn](obs_tail, tail)
        debug_assert(
            _err_small(err_head, eps) and _err_small(err_tail, eps),
            msg, ": Samples don't match!:",
            "\n\thead obs = ", self._render_samples(obs_head),
            "\n\thead exp = ", self._render_samples(head),
            "\n\thead err = ", self._render_samples(err_head),
            "\n\ttail obs = ", self._render_samples(obs_tail),
            "\n\ttail exp = ", self._render_samples(tail),
            "\n\ttail err = ", self._render_samples(err_tail),
        )

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

    fn _render_samples[samples: Int, T: Stringable & Copyable & Movable](
        self: DimensionalBuffer[dim,T],
        data: InlineArray[T,samples]
    ) -> String:
        var msg = "[ "
        @parameter
        for s in range(samples):
            @parameter
            if s > 0:
                msg += ",  "
            msg += String(data[s])
        msg += " ]"
        return msg

    fn _sample[samples: Int](self, i: Int, out data: InlineArray[T,samples]):
        data = InlineArray[T,samples](uninitialized=True)
        @parameter
        for s in range(samples):
            data[s] = self[i=i+s]

    # TEMP
    fn assert_data[err_fn: ErrFnFloat32 = err_rel](
        mut self: DimensionalBuffer[dim,Float32],
        msg: String,
        path: String,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5,
        overwrite: Bool = False
    ):

        # read the file
        var exp_data = List[Float32]()
        try:
            with open(path, "r") as f:
                var reader = FileReader(f)
                self._assert_sizes(msg, reader)
                for _ in range(self.num_elements()):
                    exp_data.append(reader.read_f32[Endian.Little]())
        except:
            debug_assert(False, msg, ": Failed to open data file")

        # look for mismatched pixels
        var mismatches = List[Self.VecD[Int]]()
        var expi = 0
        @parameter
        fn func(i: VecD[Int,dim]):
            var obs = self[i=i]
            var exp = exp_data[expi]
            expi += 1
            if not _err_small(_err[err_fn](obs, exp), eps):
                mismatches.append(i.copy())

        self.iterate[func]()

        # TEMP: need to use expi to avoid compiler bugs
        debug_assert(expi == self.num_elements(), msg, ": Iteration didn't finish")

        # report the errors
        if len(mismatches) > 0:
            var obs = self[i=mismatches[0]]
            var exp = exp_data[self._offset(mismatches[0])]
            var err = _err[err_fn](obs, exp)
            debug_assert(
                False,
                msg, ": Found ", len(mismatches), " mismatched pixels:",
                "\n\tfirst at ", mismatches[0],
                "\n\tobs = ", obs,
                "\n\texp = ", exp,
                "\n\terr = ", err,
            )

        if verbose:
            print(msg, ": sizes=", self.sizes(), " pixels ok!")

        if overwrite:
            memcpy(
                dest=self._start(),
                src=exp_data.unsafe_ptr(),
                count=self.num_elements()
            )

    # TEMP
    fn assert_data[err_fn: ErrFnFloat32 = err_rel](
        self: DimensionalBuffer[dim,ComplexFloat32],
        msg: String,
        path: String,
        *,
        verbose: Bool = False,
        eps: Float32 = 1e-5,
        overwrite: Bool = False
    ):

        # read the file
        var exp_data = List[ComplexFloat32]()
        try:
            with open(path, "r") as f:
                var reader = FileReader(f)
                self._assert_sizes(msg, reader)
                for _ in range(self.num_elements()):
                    exp_data.append(ComplexFloat32(
                        reader.read_f32[Endian.Little](),
                        reader.read_f32[Endian.Little]()
                    ))
        except:
            debug_assert(False, msg, ": Failed to open data file")

        # look for mismatched pixels
        var mismatches = List[Self.VecD[Int]]()
        var expi = 0
        @parameter
        fn func(i: VecD[Int,dim]):
            var obs = self[i=i]
            var exp = exp_data[expi]
            expi += 1
            if not _err_small(_err[err_fn](obs, exp), eps):
                mismatches.append(i.copy())

        self.iterate[func]()

        # TEMP: need to use expi to avoid compiler bugs
        debug_assert(expi == self.num_elements(), msg, ": Iteration didn't finish")

        # report the errors
        if len(mismatches) > 0:
            var obs = self[i=mismatches[0]]
            var exp = exp_data[self._offset(mismatches[0])]
            var err = _err[err_fn](obs, exp)
            debug_assert(
                False,
                msg, ": Found ", len(mismatches), " mismatched pixels:",
                "\n\tfirst at ", mismatches[0],
                "\n\tobs = ", obs,
                "\n\texp = ", exp,
                "\n\terr = ", err,
            )

        if verbose:
            print(msg, ": sizes=", self.sizes(), " pixels ok!")

        if overwrite:
            memcpy(
                dest=self._start(),
                src=exp_data.unsafe_ptr(),
                count=self.num_elements()
            )

    fn _assert_sizes(
        self,
        msg: String,
        mut reader: FileReader
    ) raises:
        var rank = reader.read_i32[Endian.Little]()
        debug_assert(
            rank == dim.rank,
            msg, ": Expected rank ", rank, " but image has rank ", dim.rank
        )
        var sizes = Self.VecD[Int](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            sizes[d] = Int(reader.read_i32[Endian.Little]())
        debug_assert(
            self.sizes() == sizes,
            msg, ": Expected sizes ", sizes, " but image has sizes ", self.sizes()
        )


# TEMP
fn err_abs(obs: Float32, exp: Float32) -> Float32:
    return abs(obs - exp)

# TEMP
fn err_rel(obs: Float32, exp: Float32) -> Float32:
    if exp == 0:
        return err_abs(obs, exp)
    else:
        return err_abs(obs, exp)/abs(exp)

alias ErrFnFloat32 = fn(Float32, Float32) -> Float32

# TEMP
fn _err[err_fn: ErrFnFloat32](obs: Float32, exp: Float32) -> Float32:
    return err_fn(obs, exp)

# TEMP
fn _err[err_fn: ErrFnFloat32](obs: ComplexFloat32, exp: ComplexFloat32) -> ComplexFloat32:
    return ComplexFloat32(
        re=err_fn(obs.re, exp.re),
        im=err_fn(obs.im, exp.im)
    )

# TEMP
fn _err[samples: Int, //, err_fn: ErrFnFloat32](
    obs: InlineArray[Float32,samples],
    exp: InlineArray[Float32,samples],
    out err: InlineArray[Float32,samples]
):
    err = InlineArray[Float32,samples](uninitialized=True)
    @parameter
    for s in range(samples):
        err[s] = _err[err_fn](obs[s], exp[s])

# TEMP
fn _err[samples: Int, //, err_fn: ErrFnFloat32](
    obs: InlineArray[ComplexFloat32,samples],
    exp: InlineArray[ComplexFloat32,samples],
    out err: InlineArray[ComplexFloat32,samples]
):
    err = InlineArray[ComplexFloat32,samples](uninitialized=True)
    @parameter
    for s in range(samples):
        err[s] = _err[err_fn](obs[s], exp[s])

fn _err_small(err: Float32, eps: Float32) -> Bool:
    return err < eps

fn _err_small(err: ComplexFloat32, eps: Float32) -> Bool:
    return _err_small(err.re, eps) and _err_small(err.im, eps)

# TEMP
fn _err_small[samples: Int, //](err: InlineArray[Float32,samples], eps: Float32) -> Bool:
    @parameter
    for s in range(samples):
        if not _err_small(err[s], eps):
            return False
    return True

# TEMP
fn _err_small[samples: Int, //](err: InlineArray[ComplexFloat32,samples], eps: Float32) -> Bool:
    @parameter
    for s in range(samples):
        if not _err_small(err[s], eps):
            return False
    return True
