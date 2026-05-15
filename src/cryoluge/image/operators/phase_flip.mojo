
from cryoluge.math import Dimension
from cryoluge.image import ComplexImage


struct PhaseFlipOperator:

    fn __init__(out self):
        # nothing to do
        pass

    fn eval[dtype: DType](
        self,
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = -v

    fn eval[dtype: DType](
        self,
        ctf: Scalar[dtype],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var sign = Int(ctf > 0) - Int(ctf < 0)
        # NOTE: mojo True casts to 1 and False to 0
        result = v*sign

    fn apply[dim: Dimension, dtype: DType](
        self,
        mut image: ComplexImage[dim,dtype]
    ):
        @parameter
        fn func(i: Vec[Int,dim]):
            image[i=i] = self.eval(image[i=i])

        image.iterate[func]()
