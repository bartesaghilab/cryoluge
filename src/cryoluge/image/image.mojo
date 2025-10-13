
from cryoluge.io import ByteBuffer


struct Image[
    dim: ImageDimension,
    dtype: DType
](Copyable, Movable):
    var _buf: DimensionalBuffer[dim,Self.PixelType]

    alias D1 = Image[ImageDimension.D1,_]
    alias D2 = Image[ImageDimension.D2,_]
    alias D3 = Image[ImageDimension.D3,_]
    alias VecD = VecD[_,dim]
    alias PixelType = Scalar[dtype]

    fn __init__(out self, sizes: Self.VecD[Int]):
        self._buf = DimensionalBuffer[dim,Self.PixelType](sizes)

    fn __init__(out self, *, sx: Int):
        self = Self(Self.VecD(x=sx))

    fn __init__(out self, *, sx: Int, sy: Int):
        self = Self(Self.VecD(x=sx, y=sy))

    fn __init__(out self, *, sx: Int, sy: Int, sz: Int):
        self = Self(Self.VecD(x=sx, y=sy, z=sz))

    fn rank(self) -> Int:
        return self._buf.rank()

    fn num_pixels(self) -> Int:
        return self._buf.num_elements()

    fn sizes(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._buf._sizes)]] Self.VecD[Int]:
        return self._buf.sizes()

    fn span(self) -> Span[Byte, MutableOrigin.cast_from[__origin_of(self._buf._buf)]]:
        return self._buf.span()

    fn __getitem__(self, *, i: Int, out v: Self.PixelType):
        v = self._buf[i=i]

    fn __getitem__(self, i: Self.VecD[Int], out v: Self.PixelType):
        v = self._buf[i]

    fn __getitem__(self, *, x: Int, out v: Self.PixelType):
        v = self._buf[x=x]

    fn __getitem__(self, *, x: Int, y: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y]

    fn __getitem__(self, *, x: Int, y: Int, z: Int, out v: Self.PixelType):
        v = self._buf[x=x, y=y, z=z]

    fn __setitem__(mut self, *, i: Int, v: Self.PixelType):
        self._buf[i=i] = v

    fn __setitem__(mut self, i: Self.VecD[Int], v: Self.PixelType):
        self._buf[i] = v

    fn __setitem__(mut self, *, x: Int, v: Self.PixelType):
        self._buf[x=x] = v

    fn __setitem__(mut self, *, x: Int, y: Int, v: Self.PixelType):
        self._buf[x=x, y=y] = v

    fn __setitem__(mut self, *, x: Int, y: Int, z: Int, v: Self.PixelType):
        self._buf[x=x, y=y, z=z] = v

    fn get(self, i: Self.VecD[Int]) -> Optional[Self.PixelType]:
        return self._buf.get(i)

    fn iterate[
        func: fn (i: Self.VecD[Int]) capturing
    ](self):
        self._buf.iterate[func]()

    # TEMP: for comparing to images in other programs
    fn assert_info[samples: Int](
        self: Image[dim,DType.float32],
        msg: String,
        sizes: Self.VecD[Int],
        head: InlineArray[Float32, samples],
        tail: InlineArray[Float32, samples],
        hash: UInt64,
        *,
        verbose: Bool = False
    ):
        self._buf.assert_info(msg, sizes, head, tail, hash, verbose=verbose)

    fn copy(self, *, center: Self.VecD[Int], padding: Self.PixelType, mut to: Self):

        var to_center = to.sizes()//2
        # NOTE: the compiler says the above variable is unsused, but it think it's wrong?
        #       the variable gets captured by the closure below, right?
        @parameter
        fn func(i: Self.VecD[Int]):
            to[i] = self.get(i + center - to_center).or_else(padding)

        to.iterate[func]()
