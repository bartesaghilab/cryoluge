
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Px


struct SwapRealSpaceQuadrantsOperator[dtype: DType, dim: Dimension]:

    var phase_shift_op: PhaseShiftOperator[dtype,dim]

    @staticmethod
    fn shifts(sizes_real: Vec[Int,dim], out result: Vec[Px[dtype],dim]):
        result = sizes_real.map_scalar[dtype]().map_unit[Px.utype]()/2

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim]
    ):
        self.phase_shift_op = PhaseShiftOperator[dtype](
            sizes_real=sizes_real.copy(),
            shifts=Self.shifts(sizes_real)
        )

    fn eval(
        self,
        *,
        f: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.phase_shift_op.eval(f=f, v=v)

    fn apply(
        self,
        mut img: FFTImage[dim,dtype]
    ):
        var coords = img.coords()

        @parameter
        fn func(i: Vec[Int,dim]):
            var f = coords.i2f(i)
            img.complex[i=i] = self.eval(f=f, v=img.complex[i=i])

        img.complex.iterate[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords
