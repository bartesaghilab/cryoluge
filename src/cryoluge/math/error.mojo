
from std.complex import ComplexScalar


def err_abs[dtype: DType](obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    return abs(obs - exp)


def err_rel[dtype: DType](obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    if exp == 0:
        return err_abs(obs, exp)
    else:
        return err_abs(obs, exp)/abs(exp)


def is_err_small[dtype: DType, //](
    err: Scalar[dtype],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    return err < eps


def is_err_small[dtype: DType, //](
    err: ComplexScalar[dtype],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    return is_err_small(err.re, eps=eps) and is_err_small(err.im, eps=eps)


def is_err_small[samples: Int, dtype: DType, //](
    err: Array[Scalar[dtype],samples],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    comptime for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


def is_err_small[samples: Int, dtype: DType, //](
    err: Array[ComplexScalar[dtype],samples],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    comptime for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


comptime ErrFn[dtype: DType] = def(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype]


def err[dtype: DType](err_fn: Some[ErrFn[dtype]], obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    return err_fn(obs, exp)


def err[dtype: DType](err_fn: Some[ErrFn[dtype]], obs: ComplexScalar[dtype], exp: ComplexScalar[dtype]) -> ComplexScalar[dtype]:
    return ComplexScalar[dtype](
        re=err_fn(obs.re, exp.re),
        im=err_fn(obs.im, exp.im)
    )


def err[samples: Int, dtype: DType, //](
    err_fn: Some[ErrFn[dtype]],
    obs: Array[Scalar[dtype],samples],
    exp: Array[Scalar[dtype],samples],
    out result: Array[Scalar[dtype],samples]
):
    result = Array[Scalar[dtype],samples](uninitialized=True)
    comptime for s in range(samples):
        result[s] = err[dtype](err_fn, obs[s], exp[s])


def err[samples: Int, dtype: DType, //](
    err_fn: Some[ErrFn[dtype]],
    obs: Array[ComplexScalar[dtype],samples],
    exp: Array[ComplexScalar[dtype],samples],
    out result: Array[ComplexScalar[dtype],samples]
):
    result = Array[ComplexScalar[dtype],samples](uninitialized=True)
    comptime for s in range(samples):
        result[s] = err[dtype](err_fn, obs[s], exp[s])
