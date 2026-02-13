
from testing import assert_equal, assert_true, assert_false
from builtin._location import __call_location, _SourceLocation

from cryoluge.lang import LexicalScope
from cryoluge.math import EulerAnglesZYZ, Matrix
from cryoluge.math.units import Deg
from cryoluge.math.error import err_abs

from cryoluge_testlib import assert_equal_angles, assert_equal_float


comptime funcs = __functions_in_module()

comptime dtype = DType.float32
comptime Deg32 = Deg[dtype]
comptime err_fn = err_abs[dtype]


def test_mat_roundtrip():

    def check(
        euler: EulerAnglesZYZ[dtype],
        *,
        exp: Optional[EulerAnglesZYZ[dtype]] = None,
        eps: Scalar[dtype] = 1e-5
    ):
        var mat = Matrix.D3[dtype](uninitialized=True)
        euler.to_matrix(mat)
        var obs = EulerAnglesZYZ(from_mat=mat)
        assert_equal_angles(obs, exp.or_else(euler), eps=eps)

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


def test_normalize():

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(5), theta=Deg32(6), phi=Deg32(7))
            .normalize(),
        5, 6, 7
    )
    
    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(-5), theta=Deg32(-6), phi=Deg32(-7))
            .normalize(),
        -5, -6, -7
    )
    
    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(-181), theta=Deg32(-182), phi=Deg32(-183))
            .normalize(),
        179, 178, 177
    )

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(181), theta=Deg32(182), phi=Deg32(183))
            .normalize(),
        -179, -178, -177
    )


def test_normalize_positive():

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(5), theta=Deg32(6), phi=Deg32(7))
            .normalize_positive(),
        5, 6, 7
    )

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(-5), theta=Deg32(-6), phi=Deg32(-7))
            .normalize_positive(),
        355, 354, 353
    )

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(-181), theta=Deg32(-182), phi=Deg32(-183))
            .normalize_positive(),
        179, 178, 177
    )

    assert_equal_euler(
        EulerAnglesZYZ(psi=Deg32(181), theta=Deg32(182), phi=Deg32(183))
            .normalize_positive(),
        181, 182, 183
    )


@always_inline
def assert_equal_euler(
    obs: EulerAnglesZYZ[dtype],
    psi: Float32,
    theta: Float32,
    phi: Float32,
    *,
    eps: Scalar[dtype] = 1e-4,
    location: Optional[_SourceLocation] = None
):
    assert_equal_float[err_fn](obs.psi.to_deg().value, psi, eps=eps, location=location.or_else(__call_location()))
    assert_equal_float[err_fn](obs.theta.to_deg().value, theta, eps=eps, location=location.or_else(__call_location()))
    assert_equal_float[err_fn](obs.phi.to_deg().value, phi, eps=eps, location=location.or_else(__call_location()))
