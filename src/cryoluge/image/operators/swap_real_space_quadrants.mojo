
from std.complex import ComplexScalar

from cryoluge.math import Vec
from cryoluge.math.units import Px
from cryoluge.image.operators.phase_shift import PhaseShiftOperator
from cryoluge.fft import FFTImage


struct SwapRealSpaceQuadrantsOperator[dim: Int, dtype: DType]
    where dim > 0 and dtype.is_floating_point():

    var phase_shift_op: PhaseShiftOperator[Self.dim,Self.dtype]

    @staticmethod
    def shifts(sizes_real: Vec[Self.dim,.int], out result: Px[Vec[Self.dim,Self.dtype]]):
        result = Px(sizes_real.map_dtype[Self.dtype]()/2)

    def __init__(
        out self,
        sizes_real: Vec[Self.dim,.int]
    ):
        self.phase_shift_op = PhaseShiftOperator[Self.dim,Self.dtype](
            sizes_real=sizes_real.copy(),
            shifts=Self.shifts(sizes_real)
        )

    def eval(
        self,
        *,
        f: Vec[Self.dim,.int],
        v: ComplexScalar[Self.dtype],
        out result: ComplexScalar[Self.dtype]
    ):
        result = self.phase_shift_op.eval(f=f, v=v)

    def apply(
        self,
        mut img: FFTImage[Self.dim,Self.dtype]
    ):
        var coords = img.coords()

        def func(i: Vec[Self.dim,.int]) {self, coords, mut img}:
            var f = coords.i2f(i)
            img.complex[i=i] = self.eval(f=f, v=img.complex[i=i])

        img.complex.pixels().iterate(func)
