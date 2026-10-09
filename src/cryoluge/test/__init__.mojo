
from std.complex import ComplexScalar
from std.testing import assert_equal, assert_true
from std.reflection import call_location, SourceLocation

from cryoluge.math import EulerAnglesZYZ, Vec
from cryoluge.math.units import Deg
from cryoluge.math.error import ErrFn, err, is_err_small


@inline(.always)
def assert_equal_buffers[
    origin_obs: Origin,
    origin_exp: Origin
](
    obs: Span[Byte,origin_obs],
    exp: Span[Byte,origin_exp],
    *,
    location: Optional[SourceLocation] = None
) raises:
    assert_equal(
        len(obs), len(exp),
        msg="Array lengths differ"
    )
    for i in range(len(exp)):
        assert_equal(
            obs[i], exp[i],
            String("Arrays differ at i=", i),
            location=location.or_else(call_location())
        )


@inline(.always)
def assert_equal_angles[dtype: DType](
    obs: EulerAnglesZYZ[dtype],
    exp: EulerAnglesZYZ[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[SourceLocation] = None
) raises where dtype.is_floating_point():

    var dists = obs.dists(exp)
    assert_true(
        dists.psi.to_deg() < eps and dists.theta.to_deg() < eps and dists.phi.to_deg() < eps,
        String("Angles mismatch!\n",
            String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
            "\tobserved: ", obs, "\n",
            "\texpected: ", exp, "\n",
            "\t   error: ", dists
        ),
        location=location.or_else(call_location())
    )


def _no_context() -> String:
    return ""


@inline(.always)
def assert_equal_float[dtype: DType, //](
    err_fn: Some[ErrFn[dtype]],
    obs: Scalar[dtype],
    exp: Scalar[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[SourceLocation] = None,
    context: Some[def () -> String] = lambda () -> String: ""
) raises:
    var err = err(err_fn, obs, exp)
    if not is_err_small(err, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t   error: ", err
            ) + context(),
            location=location.or_else(call_location())
        )

@inline(.always)
def assert_equal_float[dtype: DType, //](
    err_fn: Some[ErrFn[dtype]],
    obs: ComplexScalar[dtype],
    exp: ComplexScalar[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[SourceLocation] = None,
    context: Some[def () -> String] = lambda () -> String: ""
) raises:
    var err = err(err_fn, obs, exp)
    if not is_err_small(err, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t     err: ", err
            ) + context(),
            location=location.or_else(call_location())
        )


@inline(.always)
def assert_equal_float[
    dim: Int,
    dtype: DType,
    //
](
    err_fn: Some[ErrFn[dtype]],
    obs: Vec[dim,dtype],
    exp: Vec[dim,dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[SourceLocation] = None,
    context: Some[def () -> String] = lambda () -> String: ""
) raises where dim > 0:
    var err_sum = Scalar[dtype](0)
    comptime for d in range(dim):
        err_sum += err(err_fn, obs[d], exp[d])
    if not is_err_small(err_sum, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t     err: ", err_sum
            ) + context(),
            location=location.or_else(call_location())
        )


@inline(.always)
def assert_equal_euler[dtype: DType](
    err_fn: Some[ErrFn[dtype]],
    obs: EulerAnglesZYZ[dtype],
    psi: Deg[Scalar[dtype]],
    theta: Deg[Scalar[dtype]],
    phi: Deg[Scalar[dtype]],
    *,
    eps: Scalar[dtype] = 1e-4,
    location: Optional[SourceLocation] = None
) raises where dtype.is_floating_point():
    assert_equal_float(err_fn, obs.psi.to_deg().value, psi.value, eps=eps, location=location.or_else(call_location()))
    assert_equal_float(err_fn, obs.theta.to_deg().value, theta.value, eps=eps, location=location.or_else(call_location()))
    assert_equal_float(err_fn, obs.phi.to_deg().value, phi.value, eps=eps, location=location.or_else(call_location()))
