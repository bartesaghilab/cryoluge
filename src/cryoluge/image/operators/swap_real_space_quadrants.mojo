
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Rad


struct SwapRealSpaceQuadrantsOperator[dtype: DType, dim: Dimension]:

    var phase_shift_op: PhaseShiftOperator[dtype,dim]

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim]
    ):
        self.phase_shift_op = PhaseShiftOperator[dtype](
            sizes_real=sizes_real.copy(),
            shifts=sizes_real.map_scalar[dtype]().map_unit[Rad.utype]()/2
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
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.phase_shift_op.eval(i=i, v=v)

    fn apply(
        self,
        mut image: ComplexImage[dim,dtype]
    ):
        @parameter
        fn func(i: Vec[Int,dim]):
            image[i=i] = self.eval(i=i, v=image[i=i])

        image.iterate[func]()
