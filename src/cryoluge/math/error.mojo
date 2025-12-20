
from complex import ComplexScalar


fn err_abs[dtype: DType](obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    return abs(obs - exp)


fn err_rel[dtype: DType](obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    if exp == 0:
        return err_abs(obs, exp)
    else:
        return err_abs(obs, exp)/abs(exp)


fn is_err_small[dtype: DType](
    err: Scalar[dtype],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    return err < eps


fn is_err_small[dtype: DType](
    err: ComplexScalar[dtype],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    return is_err_small(err.re, eps=eps) and is_err_small(err.im, eps=eps)


fn is_err_small[samples: Int, //, dtype: DType](
    err: InlineArray[Scalar[dtype],samples],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    @parameter
    for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


fn is_err_small[samples: Int, //, dtype: DType](
    err: InlineArray[ComplexScalar[dtype],samples],
    *,
    eps: Scalar[dtype] = 1e-5
) -> Bool:
    @parameter
    for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


comptime ErrFn[dtype: DType] = fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype]


fn err[dtype: DType, err_fn: ErrFn[dtype]](obs: Scalar[dtype], exp: Scalar[dtype]) -> Scalar[dtype]:
    return err_fn(obs, exp)


fn err[dtype: DType, err_fn: ErrFn[dtype]](obs: ComplexScalar[dtype], exp: ComplexScalar[dtype]) -> ComplexScalar[dtype]:
    return ComplexScalar[dtype](
        re=err_fn(obs.re, exp.re),
        im=err_fn(obs.im, exp.im)
    )


fn err[samples: Int, //, dtype: DType, err_fn: ErrFn[dtype]](
    obs: InlineArray[Scalar[dtype],samples],
    exp: InlineArray[Scalar[dtype],samples],
    out result: InlineArray[Scalar[dtype],samples]
):
    result = InlineArray[Scalar[dtype],samples](uninitialized=True)
    @parameter
    for s in range(samples):
        result[s] = err[dtype,err_fn](obs[s], exp[s])


fn err[samples: Int, //, dtype: DType, err_fn: ErrFn[dtype]](
    obs: InlineArray[ComplexScalar[dtype],samples],
    exp: InlineArray[ComplexScalar[dtype],samples],
    out result: InlineArray[ComplexScalar[dtype],samples]
):
    result = InlineArray[ComplexScalar[dtype],samples](uninitialized=True)
    @parameter
    for s in range(samples):
        result[s] = err[dtype,err_fn](obs[s], exp[s])
