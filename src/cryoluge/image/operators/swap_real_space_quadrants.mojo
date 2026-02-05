
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Px


struct SwapRealSpaceQuadrantsOperator[dtype: DType, dim: Dimension]:

    var phase_shift_op: PhaseShiftOperator[dtype,dim]

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim]
    ):
        self.phase_shift_op = PhaseShiftOperator[dtype](
            sizes_real=sizes_real.copy(),
            shifts=sizes_real.map_scalar[dtype]().map_unit[Px.utype]()/2
        )

    fn eval(
        self,
        *,
        f: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.phase_shift_op.eval(f=f, v=v)

    fn eval(
        self,
        *,
        i: Vec[Int,dim],
        sizes_real: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.phase_shift_op.eval(i=i, sizes_real=sizes_real, v=v)

    fn apply(
        self,
        mut img: FFTImage[dim,dtype]
    ):
        @parameter
        fn func(i: Vec[Int,dim]):
            img.complex[i=i] = self.eval(i=i, sizes_real=img.sizes_real, v=img.complex[i=i])

        img.complex.iterate[func]()
