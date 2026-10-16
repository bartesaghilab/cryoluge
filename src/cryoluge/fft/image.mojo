
from std.math import pi, cos, sin, floor

from cryoluge.math import Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage
from cryoluge.fft.coords import FFTCoords


struct FFTImage[
    dim: Int,
    dtype: DType
](Copyable, Movable)
    where dim > 0:
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: Vec[Self.dim,.int]
    var complex: ComplexImage[Self.dim, Self.dtype]

    comptime Vec = ComplexImage[Self.dim, Self.dtype].Vec
    comptime PixelType = ComplexImage[Self.dim, Self.dtype].PixelType
    comptime PixelVec = ComplexImage[Self.dim, Self.dtype].PixelVec
    comptime ScalarType = ComplexImage[Self.dim, Self.dtype].ScalarType
    comptime ScalarVec = ComplexImage[Self.dim, Self.dtype].ScalarVec

    def __init__(
        out self,
        sizes_real: Self.Vec[.int],
        *,
        uninitialized: Bool,
        alignment: Optional[Int] = None
    ):
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[Self.dim,Self.dtype](
            fft_coords.sizes_fourier(),
            uninitialized=True,
            alignment=alignment
        )

    def coords(self) -> FFTCoords[Self.dim]:
        return FFTCoords(self.sizes_real)

    def crop(
        self,
        *,
        mut to: Self,
        out_of_range: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)
    ):
        ref dst = to
        ref src = self

        var coords_dst = dst.coords()
        var coords_src = src.coords()

        # sample into the dst image
        def func(i_dst: Self.Vec[.int]) {src, coords_src, mut dst, coords_dst, out_of_range}:
            var i_src = coords_src.maybe_f2i(coords_dst.i2f(i_dst))
            if i_src is not None:
                dst.complex[i_dst] = src.complex[i_src.value()]
            else:
                dst.complex[i_dst] = out_of_range

        dst.complex.pixels().iterate(func)

    def find(
        self,
        *,
        f: Self.Vec[.int],
        out v: Optional[ComplexScalar[Self.dtype]]
    ):
        var i = self.coords().maybe_f2i(f)
        if i is None:
            v = None
        else:
            v = self.complex.get(i.value())
            if v is not None and self.coords().needs_conjugation(f=f):
                v = v.value().conj()

    def get[*, or_else: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)](
        self,
        *,
        f: Self.Vec[.int],
        out v: ComplexScalar[Self.dtype]
    ):
        # NOTE: manually inlining all the FFTCoords operations here and simplifying
        #       can't seem to beat the compiler's optimizations, so go mojo! =D
        var i = self.coords().maybe_f2i(f)
        if i is None:
            v = materialize[or_else]()
        else:
            v = self.complex.get[or_else=or_else](i.value())
            if self.coords().needs_conjugation(f=f):
                v = v.conj()

    def get[
        *,
        or_else: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        dtype_coords: DType = DType.float32
    ](
        self,
        *,
        f_lerp: Vec[Self.dim,dtype_coords],
        out pixel: ComplexScalar[Self.dtype]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Self.Vec[.int](uninitialized=True)
        var dists = Vec[Self.dim,dtype_coords](uninitialized=True)
        comptime for d in range(Self.dim):
            var floor = floor(f_lerp[d])
            start[d] = Int(floor)
            dists[d] = f_lerp[d] - floor

        pixel = ComplexScalar[Self.dtype](0, 0)

        comptime for delta in Delta[Self.dim,dtype_coords].build():
            var f_sample = start + materialize[delta.pos]()
            var v = self.get[or_else=or_else](f=f_sample)
            var weight = Scalar[Self.dtype]((dists*materialize[delta.dir]() + materialize[delta.pos_f]()).product())
            pixel = pixel + v*weight


@fieldwise_init
struct Delta[
    dim: Int,
    dtype: DType
](
    Copyable,
    Movable,
    Equatable,
    Writable
):
    var d: Int
    var pos: Vec[Self.dim, .int]
    """[0], unless flipped, then 1@d."""
    var pos_f: Vec[Self.dim, Self.dtype]
    """[1], unless flipped, then 0@d."""
    var dir: Vec[Self.dim, Self.dtype]
    """[-1], unless flipped, then 1@d."""

    @staticmethod
    def build(out deltas: List[Self]):
        deltas = [Self()]
        for d in range(Self.dim):
            for i in range(len(deltas)):
                var delta = deltas[i].copy()
                delta.d = d
                delta.flip()
                deltas.append(delta^)

    @staticmethod
    def dir_pos[num_samples: Int](
        deltas: List[Self],
        out dir_pos: Tuple[SIMD[Self.dtype,num_samples], SIMD[Self.dtype,num_samples]]
    ):
        var dir = SIMD[Self.dtype,num_samples](0)
        var pos = SIMD[Self.dtype,num_samples](0)
        for s in range(num_samples):
            ref delta = deltas[s]
            dir[s] = delta.dir[delta.d]
            pos[s] = Scalar[Self.dtype](delta.pos[delta.d])
        dir_pos = (dir, pos)

    def __init__(out self):
        self.d = 0
        self.pos = Vec[Self.dim, .int](fill=0)
        self.pos_f = Vec[Self.dim, Self.dtype](fill=1)
        self.dir = Vec[Self.dim, Self.dtype](fill=-1)

    def flip(mut self):
        self.pos[self.d] = 1
        self.pos_f[self.d] = 0
        self.dir[self.d] = 1

    def __eq__(self, rhs: Self) -> Bool:
        return self.pos == rhs.pos

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Delta[pos=", self.pos, ", pos_f=", self.pos_f, ", dir=", self.dir, "]")
