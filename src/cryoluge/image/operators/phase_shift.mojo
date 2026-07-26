
from math import sin, cos

from cryoluge.math import Vec
from cryoluge.math.units import Px, Rad, pi
from cryoluge.fft import FFTCoords, FFTImage


struct PhaseShiftOperator[dtype: DType, dim: Int](
    Copyable,
    Movable
):
    var _shifts: Vec[dim,Rad[dtype]]

    fn __init__(
        out self,
        sizes_real: Vec[dim,Int],
        shifts: Vec[dim,Px[dtype]]
    ):
        # convert shifts from Px to Rad
        self._shifts = shifts.map_value()*2*pi[dtype].value/sizes_real.map_scalar[dtype]()
            .map_unit[Rad.utype]()

    @always_inline
    fn get[width: Int](
        self,
        *,
        f: Vec[dim,SIMD[dtype,width]],
        out result: ComplexSIMD[dtype,width]
    ):
        var phases = SIMD[dtype,width](0)
        @parameter
        for d in range(dim):
            phases += f[d]*self._shifts[d].value
        result = ComplexSIMD[dtype,width](re=cos(phases), im=-sin(phases))

    @always_inline
    fn eval[width: Int](
        self,
        *,
        f: Vec[dim,SIMD[dtype,width]],
        v: ComplexSIMD[dtype,width],
        out result: ComplexSIMD[dtype,width]
    ):
        result = v*self.get(f=f)

    @always_inline
    fn eval(
        self,
        *,
        f: Vec[dim,Int],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = self.eval(f=f.map_scalar[dtype](), v=v)

    fn apply(
        self,
        mut img: FFTImage[dim,dtype]
    ):
        var coords = img.coords()

        @parameter
        fn func(i: Vec[dim,Int]):
            var f = coords.i2f(i)
            img.complex[i=i] = self.eval(f=f, v=img.complex[i=i])

        img.complex.iterate[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords