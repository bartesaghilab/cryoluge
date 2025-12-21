
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Rad, pi
from cryoluge.fft import FFTCoords


struct PhaseShiftOperator[dtype: DType, dim: Dimension](
    Copyable,
    Movable
):

    var sizes_real: Vec[Int,dim]
    var _shifts_2pi: Vec[Rad[dtype],dim]

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim],
        shifts: Vec[Rad[dtype],dim]
    ):
        self.sizes_real = sizes_real.copy()
        self._shifts_2pi = shifts.copy()*2*pi[dtype].value

    fn eval(
        self,
        *,
        freqs: Vec[Scalar[dtype],dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var phase = (self._shifts_2pi*freqs).sum()
        result = v*ComplexScalar[dtype](re=phase.cos(), im=-phase.sin())

    fn eval(
        self,
        *,
        f: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var freqs = FFTCoords(self.sizes_real).freqs[dtype](f=f)
        result = self.eval(freqs=freqs, v=v)

    fn eval(
        self,
        *,
        f: Vec[Scalar[dtype],dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var freqs = FFTCoords(self.sizes_real).freqs[dtype](f=f)
        result = self.eval(freqs=freqs, v=v)

    fn eval(
        self,
        *,
        i: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var f = FFTCoords(self.sizes_real).i2f(i)
        result = self.eval(f=f, v=v)

    fn apply(
        self,
        mut image: ComplexImage[dim,dtype]
    ):
        @parameter
        fn func(i: Vec[Int,dim]):
            image[i=i] = self.eval(i=i, v=image[i=i])

        image.iterate[func]()
