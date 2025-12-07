
from math import pi, cos, sin, floor

from cryoluge.math import Dimension, Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage
from cryoluge.ctf import CTF


struct FFTImage[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: Vec[Int,dim]
    var complex: ComplexImage[dim,dtype]

    comptime D1 = FFTImage[Dimension.D1,_]
    comptime D2 = FFTImage[Dimension.D2,_]
    comptime D3 = FFTImage[Dimension.D3,_]
    comptime Vec = ComplexImage[dim,dtype].Vec
    comptime PixelType = ComplexImage[dim,dtype].PixelType
    comptime PixelVec = ComplexImage[dim,dtype].PixelVec
    comptime ScalarType = ComplexImage[dim,dtype].ScalarType
    comptime ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: Self.Vec[Int], *, alignment: Optional[Int] = None):
        """WARNING: produces un-initialized memory."""
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[dim,dtype](fft_coords.sizes_fourier(), alignment=alignment)

    fn __init__(out self, *, of: Image[dim,dtype], alignment: Optional[Int] = None):
        """WARNING: produces un-initialized memory."""
        ref real = of
        self = Self(real.sizes(), alignment=alignment)

    fn coords(self) -> FFTCoords[dim, origin=origin_of(self.sizes_real)]:
        return FFTCoords(self.sizes_real)

    fn crop(self, *, mut to: Self):
        ref dst = to

        # make sure the destination image is smaller (or the same size) as this one
        @parameter
        for d in range(dim.rank):
            debug_assert(
                dst.sizes_real[d] <= self.sizes_real[d],
                "Crop destination real sizes ", dst.sizes_real,
                " must be smaller (or same size) as this source real sizes ", self.sizes_real
            )

        # sample into the dst image
        @parameter
        fn sample(i: Self.Vec[Int]):
            dst.complex[i] = self.complex[self.coords().f2i(dst.coords().i2f(i))]

        dst.complex.iterate[sample]()

        _ = dst  # TEMP: need to extend lifetime of ref to avoid compiler bug

    fn get[*, or_else: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)](
        self,
        *,
        f: Self.Vec[Int],
        out v: ComplexScalar[dtype]
    ):
        # NOTE: manually inlining all the FFTCoords operations here and simplifying
        #       can't seem to beat the compiler's optimizations, so go mojo! =D
        if self.coords().needs_conjugation(f=f):
            var i = self.coords().f2i_flipped(f)
            v = self.complex.get[or_else=or_else](i).conj()
        else:
            var i = self.coords().f2i(f)
            v = self.complex.get[or_else=or_else](i)

    fn get[*, or_else: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)](
        self: FFTImage[dim,dtype],
        *,
        f_lerp: Self.Vec[Float32],
        out pixel: ComplexScalar[dtype]
    ):
        # discretize the frequency coordinates, and keep track of distances
        var start = Vec[Int,dim](uninitialized=True)
        var dists = Vec[Float32,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            var floor = floor(f_lerp[d])
            start[d] = Int(floor)
            dists[d] = f_lerp[d] - floor

        # build the multi-dimensional delta vectors (at compile-time)
        fn build_deltas(out deltas: List[Delta[dim]]):
            deltas = [Delta[dim]()]
            for d in range(dim.rank):
                for i in range(len(deltas)):
                    var delta = deltas[i].copy()
                    delta.flip(d)
                    deltas.append(delta^)

        pixel = ComplexScalar[dtype](0, 0)

        @parameter
        for delta in build_deltas():
            var f_sample = start + materialize[delta.pos]()
            var v = self.get[or_else=or_else](f=f_sample)
            var weight = Scalar[dtype]((dists*materialize[delta.dir]() + materialize[delta.pos_f]()).product())
            pixel = pixel + v*weight


@fieldwise_init
struct Delta[dim: Dimension](
    Copyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var pos: Vec[Int,dim]
    var pos_f: Vec[Float32,dim]
    var dir: Vec[Float32,dim]

    fn __init__(out self):
        self.pos = Vec[Int,dim](fill=0)
        self.pos_f = Vec[Float32,dim](fill=1)
        self.dir = Vec[Float32,dim](fill=-1)

    fn flip(mut self, d: Int):
        self.pos[d] = 1
        self.pos_f[d] = 0
        self.dir[d] = 1

    fn __eq__(self, rhs: Self) -> Bool:
        return self.pos == rhs.pos

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Delta[pos=", self.pos, ", pos_f=", self.pos_f, ", dir=", self.dir, "]")

    fn __str__(self) -> String:
        return String.write(self)
