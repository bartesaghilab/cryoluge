
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Rad, pi
from cryoluge.fft import FFTCoords


struct PhaseShiftOperator[dtype: DType, dim: Dimension](
    Copyable,
    Movable
):

    var sizes_real: Vec[Int,dim]
    var shifts: Vec[Rad[dtype],dim]

    fn __init__(
        out self,
        sizes_real: Vec[Int,dim],
        shifts: Vec[Rad[dtype],dim]
    ):
        self.sizes_real = sizes_real.copy()
        self.shifts = shifts.copy()

    fn eval(
        self,
        *,
        f: Vec[Int,dim],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        # TODO: this code uses more library fns, but doesn't quite match csp1, due to different roundoff error
        #       ironically, it would match better if we could do the division in freqs(),
        #       but that would cause different roundoff error in a *different* location =(
        # var freqs = FFTCoords(self.sizes_real).freqs[dtype](f=f)
        # var phase = 0 - (self.shifts*2*pi[dtype]*freqs).sum()

        # TEMP: this code matches csp1, but re-implements frequency calculations
        var freq = f.map_scalar[dtype]()
        var sizes_real = self.sizes_real.map_scalar[dtype]()
        var phase = 0 - (freq*self.shifts*2*pi[dtype].value/sizes_real).sum()

        result = v*ComplexScalar[dtype](re=phase.cos(), im=phase.sin())

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
