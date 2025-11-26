

from cryoluge.math import Dimension, Vec
from cryoluge.fft import FFTCoords
from cryoluge.image import ComplexImage
from cryoluge.image.analysis import FourierShells


struct WhitenOperator[
    dim: Dimension,
    dtype: DType,
    sum_dtype: DType=dtype
]:
    var real_sizes: Vec[Int,dim]
    var shells: FourierShells[dim]
    var res_limit: Int
    var _sum: List[Scalar[sum_dtype]]
    var _count: List[Int]

    fn __init__(
        out self,
        real_sizes: Vec[Int,dim],
        shells: FourierShells[dim],
        *,
        res_limit: Optional[Int] = None
    ):
        self.real_sizes = real_sizes.copy()
        self.shells = shells.copy()
        self.res_limit = res_limit.or_else(shells.count)
        self._sum = List[Scalar[sum_dtype]](length=len(shells), fill=0)
        self._count = List[Int](length=len(shells), fill=0)

    fn reset(mut self):
        for shelli in range(len(self._sum)):
            self._sum[shelli] = 0
            self._count[shelli] = 0

    fn collect_statistics(mut self, *, f: Vec[Int,dim], v: ComplexScalar[dtype]):
        var dist2 = v.squared_norm()
        if dist2 > 0:
            var shelli = self.shells.shelli[dtype](f=f)
            if shelli <= self.res_limit:
                self._sum[shelli] += Scalar[sum_dtype](dist2)
                self._count[shelli] += 1

    fn normalize(mut self):
        for shelli in range(len(self._sum)):
            if self._count[shelli] > 0:
                self._sum[shelli] = sqrt(self._sum[shelli]/Scalar[sum_dtype](self._count[shelli]))

    fn eval(self, *, f: Vec[Int,dim], v: ComplexScalar[dtype], out result: ComplexScalar[dtype]):
        var shelli = self.shells.shelli[dtype](f=f)
        if shelli <= self.res_limit and self._count[shelli] > 0:
            var sum = Scalar[dtype](self._sum[shelli])
            result = ComplexScalar[dtype](re=v.re/sum, im=v.im/sum)
        else:
            result = ComplexScalar[dtype](0, 0)

    fn apply(mut self, mut image: ComplexImage[dim,dtype]):

        self.reset()

        @parameter
        fn func1(i: Vec[Int,dim]):
            var v = image[i=i]
            var f = FFTCoords(self.real_sizes).i2f(i)
            self.collect_statistics(f=f, v=v)

        image.iterate[func1]()

        self.normalize()

        @parameter
        fn func2(i: Vec[Int,dim]):
            var v = image[i=i]
            var f = FFTCoords(self.real_sizes).i2f(i)
            image[i=i] = self.eval(f=f, v=v)

        image.iterate[func2]()
