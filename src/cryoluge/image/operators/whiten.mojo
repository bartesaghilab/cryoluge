
from std.math import sqrt
from std.complex import ComplexScalar

from cryoluge.math import Vec
from cryoluge.fft import FFTImage
from cryoluge.image import ComplexImage
from cryoluge.image.analysis import FourierShells


struct WhitenOperator[
    dtype: DType,
    dim: Int,
    sum_dtype: DType,
    origin_shells: Origin[mut=False]
](
    Movable
) where dim > 0:

    var stats: WhitenStats[Self.sum_dtype,Self.dim,Self.origin_shells]
    var _factor: List[Scalar[Self.dtype]]

    def __init__(
        out self,
        var stats: WhitenStats[Self.sum_dtype,Self.dim,Self.origin_shells]
    ):
        self.stats = stats^

        # normalize the statistics
        self._factor = List[Scalar[Self.dtype]](capacity=len(self.stats.shells[]))
        for shelli in range(len(self.stats.shells[])):
            var sum = self.stats._sum[shelli]
            var count = self.stats._count[shelli]
            var factor: Scalar[Self.dtype]
            if shelli > self.stats.res_limit:
                factor = 0
            elif count > 0:
                factor = Scalar[Self.dtype](1/sqrt(sum/Scalar[Self.sum_dtype](count)))
            else:
                factor = 1
            self._factor.append(factor)

    def eval(
        self,
        *,
        freq_norm: Scalar[Self.dtype],
        v: ComplexScalar[Self.dtype],
        out result: ComplexScalar[Self.dtype]
    ):
        var shelli = self.stats.shells[].shelli(freq_norm=freq_norm)
        result = v*self._factor[shelli]

    def apply(self, mut image: FFTImage[Self.dim,Self.dtype]):

        var coords = image.coords()

        def func(i: Vec[Self.dim,.int]) {self, mut image, coords}:
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var f_norm = coords.f_norm[Self.dtype](f=f)
            image.complex[i=i] = self.eval(freq_norm=f_norm.len(), v=v)

        image.complex.pixels().iterate(func)


struct WhitenStats[
    sum_dtype: DType,
    dim: Int,
    origin_shells: Origin[mut=False]
](
    Movable
) where dim > 0:

    var shells: Pointer[FourierShells[Self.dim],Self.origin_shells]
    var res_limit: Int
    var _sum: List[Scalar[Self.sum_dtype]]
    var _count: List[Int]

    def __init__(
        out self,
        ref [Self.origin_shells] shells: FourierShells[Self.dim],
        *,
        res_limit: Optional[Int] = None
    ):
        self.shells = Pointer(to=shells)
        self.res_limit = res_limit.or_else(self.shells[].count_at_unity)
        self._sum = List[Scalar[Self.sum_dtype]](length=len(shells), fill=0)
        self._count = List[Int](length=len(shells), fill=0)

    def eval[dtype: DType](mut self, *, freq_norm2: Scalar[dtype], v: ComplexScalar[dtype]):
        var dist2 = v.squared_norm()
        if dist2 > 0:
            var shelli = self.shells[].shelli(freq_norm2=freq_norm2)
            if shelli <= self.res_limit:
                self._sum[shelli] += Scalar[Self.sum_dtype](dist2)
                self._count[shelli] += 1

    def apply[dtype: DType](mut self, mut image: FFTImage[Self.dim,dtype]):

        var coords = image.coords()
        
        def func(i: Vec[Self.dim,.int]) {mut self, image, coords}:
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var freq_norm2 = coords.f_norm[dtype](f=f).len2()
            self.eval(freq_norm2=freq_norm2, v=v)

        image.complex.pixels().iterate(func)
