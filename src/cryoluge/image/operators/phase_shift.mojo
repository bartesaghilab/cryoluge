
from math import sin, cos

from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Px, Rad, pi
from cryoluge.fft import FFTCoords, FFTImage


struct PhaseShiftOperator[dtype: DType, dim: Dimension](
    Copyable,
    Movable
):
    var _shifts: Vec[Rad[dtype],dim]

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim],
        shifts: Vec[Px[dtype],dim]
    ):
        # convert shifts from Px to Rad
        self._shifts = shifts.map_value()*2*pi[dtype].value/sizes_real.map_scalar[dtype]()
            .map_unit[Rad.utype]()

    fn eval(
        self,
        *,
        f: Vec[Scalar[dtype],dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var phase = self._shifts.inner_product(f)
        result = v*ComplexScalar[dtype](re=phase.cos(), im=-phase.sin())

    fn eval[width: Int](
        self: PhaseShiftOperator[dtype,Dimension.D2],
        *,
        f: Vec.D2[SIMD[dtype,width]],
        v: ComplexSIMD[dtype,width],
        out result: ComplexSIMD[dtype,width]
    ):
        var phases = SIMD[dtype,width](0)
        @parameter
        for d in range(dim.rank):
            phases += f[d]*self._shifts[d].value
        result = v*ComplexSIMD[dtype,width](re=cos(phases), im=-sin(phases))

    fn eval(
        self,
        *,
        f: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.eval(f=f.map_scalar[dtype](), v=v)

    fn eval(
        self,
        *,
        i: Vec[Int,dim],
        sizes_real: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var f = FFTCoords(sizes_real).i2f(i)
        result = self.eval(f=f, v=v)

    fn apply(
        self,
        mut img: FFTImage[dim,dtype]
    ):
        @parameter
        fn func(i: Vec[Int,dim]):
            img.complex[i=i] = self.eval(i=i, sizes_real=img.sizes_real, v=img.complex[i=i])

        img.complex.iterate[func]()
