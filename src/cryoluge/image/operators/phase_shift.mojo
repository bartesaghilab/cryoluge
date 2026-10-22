
from std.math import sin, cos
from std.complex import ComplexSIMD

from cryoluge.math import Vec
from cryoluge.math.units import Px, Rad, pi
from cryoluge.fft import FFTCoords, FFTImage


struct PhaseShiftOperator[dim: Int, dtype: DType](
    Copyable,
    Movable
) where dim > 0 and dtype.is_floating_point():

    var _shifts: Rad[Self.VecF[]]

    comptime VecI[slen: SIMDLength = 1] = Vec[Self.dim, .int, slen]
    comptime VecF[slen: SIMDLength = 1] = Vec[Self.dim, Self.dtype, slen]
    comptime Complex[slen: SIMDLength = 1] = ComplexSIMD[Self.dtype, slen]

    def __init__(
        out self,
        sizes_real: Self.VecI[],
        shifts: Px[Self.VecF[]]
    ):
        # convert shifts from Px to Rad
        self._shifts = shifts.value*2*pi[Self.dtype].value/sizes_real.map_dtype[Self.dtype]()

    @inline(.always)
    def get[slen: SIMDLength](
        self,
        *,
        f: Self.VecF[slen],
        out result: Self.Complex[slen]
    ):
        var phases = SIMD[Self.dtype,slen](0)
        comptime for d in range(Self.dim):
            phases += f[d]*self._shifts.value[d]
        result = ComplexSIMD[Self.dtype,slen](re=cos(phases), im=-sin(phases))

    @inline(.always)
    def eval[slen: SIMDLength](
        self,
        *,
        f: Self.VecF[slen],
        v: Self.Complex[slen],
        out result: Self.Complex[slen]
    ):
        result = v*self.get(f=f)

    @inline(.always)
    def eval(
        self,
        *,
        f: Self.VecI[],
        v: Self.Complex[],
        out result: Self.Complex[]
    ):
        result = self.eval(f=f.map_dtype[Self.dtype](), v=v)

    def apply(
        self,
        mut img: FFTImage[Self.dim,Self.dtype]
    ):
        var coords = img.coords()

        def func(i: Self.VecI[]) {self, mut img, coords}:
            var f = coords.i2f(i)
            img.complex[i=i] = self.eval(f=f, v=img.complex[i=i])

        img.complex.pixels().iterate(func)
