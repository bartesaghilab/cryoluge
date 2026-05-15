

from cryoluge.math import Dimension, Vec
from cryoluge.fft import FFTImage
from cryoluge.image import ComplexImage
from cryoluge.image.analysis import FourierShells


struct WhitenOperator[
    dtype: DType,
    dim: Dimension,
    sum_dtype: DType,
    origin_shells: Origin[mut=False]
](
    Movable
):
    var stats: WhitenStats[sum_dtype,dim,origin_shells]
    var _factor: List[Scalar[dtype]]

    fn __init__(
        out self,
        var stats: WhitenStats[sum_dtype,dim,origin_shells]
    ):
        self.stats = stats^

        # normalize the statistics
        self._factor = List[Scalar[dtype]](capacity=len(self.stats.shells[]))
        for shelli in range(len(self.stats.shells[])):
            var sum = self.stats._sum[shelli]
            var count = self.stats._count[shelli]
            var factor: Scalar[dtype]
            if shelli > self.stats.res_limit:
                factor = 0
            elif count > 0:
                factor = Scalar[dtype](1/sqrt(sum/Scalar[sum_dtype](count)))
            else:
                factor = 1
            self._factor.append(factor)

    fn eval(
        self,
        *,
        freq: Scalar[dtype],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var shelli = self.stats.shells[].shelli(freq=freq)
        result = v*self._factor[shelli]

    fn apply(self, mut image: FFTImage[dim,dtype]):

        var coords = image.coords()

        @parameter
        fn func(i: Vec[Int,dim]):
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var freq = sqrt(coords.freqs[dtype](f=f).len2())
            image.complex[i=i] = self.eval(freq=freq, v=v)

        image.complex.iterate[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords


struct WhitenStats[
    sum_dtype: DType,
    dim: Dimension,
    origin_shells: Origin[mut=False]
](
    Movable
):
    var shells: Pointer[FourierShells[dim],origin_shells]
    var res_limit: Int
    var _sum: List[Scalar[sum_dtype]]
    var _count: List[Int]

    fn __init__(
        out self,
        ref [origin_shells] shells: FourierShells[dim],
        *,
        res_limit: Optional[Int] = None
    ):
        self.shells = Pointer(to=shells)
        self.res_limit = res_limit.or_else(self.shells[].count)
        self._sum = List[Scalar[sum_dtype]](length=len(shells), fill=0)
        self._count = List[Int](length=len(shells), fill=0)

    fn eval[dtype: DType](mut self, *, freq2: Scalar[dtype], v: ComplexScalar[dtype]):
        var dist2 = v.squared_norm()
        if dist2 > 0:
            var shelli = self.shells[].shelli(freq2=freq2)
            if shelli <= self.res_limit:
                self._sum[shelli] += Scalar[sum_dtype](dist2)
                self._count[shelli] += 1

    fn apply[dtype: DType](mut self, mut image: FFTImage[dim,dtype]):

        var coords = image.coords()
        
        @parameter
        fn func(i: Vec[Int,dim]):
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var freq2 = coords.freqs[dtype](f=f).len2()
            self.eval(freq2=freq2, v=v)

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords

        image.complex.iterate[func]()
