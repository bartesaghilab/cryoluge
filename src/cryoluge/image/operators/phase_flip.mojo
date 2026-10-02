
from cryoluge.image import ComplexImage


struct PhaseFlipOperator:

    def __init__(out self):
        # nothing to do
        pass

    def eval[dtype: DType](
        self,
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = -v

    def eval[dtype: DType](
        self,
        ctf: Scalar[dtype],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        var sign = Int(ctf > 0) - Int(ctf < 0)
        # NOTE: mojo True casts to 1 and False to 0
        result = v*sign

    def apply[dim: Int, dtype: DType](
        self,
        mut image: ComplexImage[dim,dtype]
    ):
        @parameter
        def func(i: Vec[dim,Int]):
            image[i=i] = self.eval(image[i=i])

        image.iterate[func]()
