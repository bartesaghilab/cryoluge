
from complex import ComplexScalar

from cryoluge.math import EulerAnglesZYZ, Vec
from cryoluge.math.error import ErrFn, err, is_err_small

from testing import assert_equal, assert_true
from builtin._location import __call_location, _SourceLocation


@always_inline
def assert_equal_buffers(
    obs: Span[Byte],
    exp: Span[Byte],
    *,
    location: Optional[_SourceLocation] = None
):
    assert_equal(
        len(obs), len(exp),
        msg="Array lengths differ"
    )
    for i in range(len(exp)):
        assert_equal(
            obs[i], exp[i],
            String("Arrays differ at i=", i),
            location=location.or_else(__call_location())
        )


@always_inline
def assert_equal_angles[dtype: DType](
    obs: EulerAnglesZYZ[dtype],
    exp: EulerAnglesZYZ[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[_SourceLocation] = None
):
    var dists = obs.dists(exp)
    assert_true(
        dists.psi.to_deg() < eps and dists.theta.to_deg() < eps and dists.phi.to_deg() < eps,
        String("Angles mismatch!\n",
            String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
            "\tobserved: ", obs, "\n",
            "\texpected: ", exp, "\n",
            "\t   error: ", dists
        ),
        location=location.or_else(__call_location())
    )


@parameter
fn _no_context() -> String:
    return ""


@always_inline
def assert_equal_float[
    dtype: DType,
    //,
    err_fn: ErrFn[dtype],
    context: fn () capturing -> String = _no_context
](
    obs: Scalar[dtype],
    exp: Scalar[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[_SourceLocation] = None
):
    var err = err[dtype,err_fn](obs, exp)
    if not is_err_small(err, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t   error: ", err
            ) + context(),
            location=location.or_else(__call_location())
        )

@always_inline
def assert_equal_float[
    dtype: DType,
    //,
    err_fn: ErrFn[dtype],
    context: fn () capturing -> String = _no_context
](
    obs: ComplexScalar[dtype],
    exp: ComplexScalar[dtype],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[_SourceLocation] = None
):
    var err = err[dtype,err_fn](obs, exp)
    if not is_err_small(err, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t     err: ", err
            ) + context(),
            location=location.or_else(__call_location())
        )


@always_inline
def assert_equal_float[
    dim: Int,
    dtype: DType,
    //,
    err_fn: ErrFn[dtype],
    context: fn () capturing -> String = _no_context
](
    obs: Vec[dim,Scalar[dtype]],
    exp: Vec[dim,Scalar[dtype]],
    msg: Optional[String] = None,
    *,
    eps: Scalar[dtype] = 1e-5,
    location: Optional[_SourceLocation] = None
):
    var err_sum = Scalar[dtype](0)
    @parameter
    for d in range(dim):
        err_sum += err[dtype,err_fn](obs[d], exp[d])
    if not is_err_small(err_sum, eps=eps):
        assert_true(
            False,
            String("Floats mismatch!\n",
                String("\t     msg: ", msg.value(), "\n") if msg is not None else "",
                "\tobserved: ", obs, "\n",
                "\texpected: ", exp, "\n",
                "\t     err: ", err_sum
            ) + context(),
            location=location.or_else(__call_location())
        )
