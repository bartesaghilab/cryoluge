
from testing import assert_equal, assert_true, assert_false

from cryoluge.lang import LexicalScope
from cryoluge.math import EulerAnglesZYZ, Matrix
from cryoluge.math.units import Deg

from cryoluge_testlib import assert_equal_angles


comptime funcs = __functions_in_module()


def test_mat_roundtrip():

    def check[dtype: DType](
        euler: EulerAnglesZYZ[dtype],
        *,
        exp: Optional[EulerAnglesZYZ[dtype]] = None,
        eps: Scalar[dtype] = 1e-5
    ):
        var mat = Matrix.D3[dtype](uninitialized=True)
        euler.to_matrix(mat)
        var obs = EulerAnglesZYZ(from_mat=mat)
        assert_equal_angles(obs, exp.or_else(euler), eps=eps)

    comptime Deg32 = Deg[DType.float32]

    check(EulerAnglesZYZ(psi=Deg32(0), theta=Deg32(0), phi=Deg32(0)))

    check(EulerAnglesZYZ[DType.float32](psi=Deg32(1), theta=Deg32(0), phi=Deg32(0)))
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(0), theta=Deg32(1), phi=Deg32(0)),
        eps=1e-4)
        # NOTE: ZYZ Euler angle decomposition has low precision around theta=0
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(0), theta=Deg32(0), phi=Deg32(1)),
        exp=EulerAnglesZYZ[DType.float32](psi=Deg32(1), theta=Deg32(0), phi=Deg32(0)))
        # NOTE: when psi and theta are zero, a phi=n rotation is equivalent to a psi=n rotation
    
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(30), theta=Deg32(0), phi=Deg32(0)))
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(0), theta=Deg32(30), phi=Deg32(0)))
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(0), theta=Deg32(0), phi=Deg32(30)),
        exp=EulerAnglesZYZ[DType.float32](psi=Deg32(30), theta=Deg32(0), phi=Deg32(0)))

    # real-world example
    check(EulerAnglesZYZ[DType.float32](psi=Deg32(71.149391), theta=Deg32(11.956297), phi=Deg32(59.530159)))
