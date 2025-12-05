
from complex import ComplexFloat32


# TODO: Float64 overloads?


fn err_abs(obs: Float32, exp: Float32) -> Float32:
    return abs(obs - exp)


fn err_rel(obs: Float32, exp: Float32) -> Float32:
    if exp == 0:
        return err_abs(obs, exp)
    else:
        return err_abs(obs, exp)/abs(exp)


fn is_err_small(
    err: Float32,
    *,
    eps: Float32 = 1e-5
) -> Bool:
    return err < eps


fn is_err_small(
    err: ComplexFloat32,
    *,
    eps: Float32 = 1e-5
) -> Bool:
    return is_err_small(err.re, eps=eps) and is_err_small(err.im, eps=eps)


fn is_err_small[samples: Int, //](
    err: InlineArray[Float32,samples],
    *,
    eps: Float32 = 1e-5
) -> Bool:
    @parameter
    for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


fn is_err_small[samples: Int, //](
    err: InlineArray[ComplexFloat32,samples],
    *,
    eps: Float32 = 1e-5
) -> Bool:
    @parameter
    for s in range(samples):
        if not is_err_small(err[s], eps=eps):
            return False
    return True


comptime ErrFnFloat32 = fn(Float32, Float32) -> Float32


fn err[err_fn: ErrFnFloat32](obs: Float32, exp: Float32) -> Float32:
    return err_fn(obs, exp)


fn err[err_fn: ErrFnFloat32](obs: ComplexFloat32, exp: ComplexFloat32) -> ComplexFloat32:
    return ComplexFloat32(
        re=err_fn(obs.re, exp.re),
        im=err_fn(obs.im, exp.im)
    )


fn err[samples: Int, //, err_fn: ErrFnFloat32](
    obs: InlineArray[Float32,samples],
    exp: InlineArray[Float32,samples],
    out result: InlineArray[Float32,samples]
):
    result = InlineArray[Float32,samples](uninitialized=True)
    @parameter
    for s in range(samples):
        result[s] = err[err_fn](obs[s], exp[s])


fn err[samples: Int, //, err_fn: ErrFnFloat32](
    obs: InlineArray[ComplexFloat32,samples],
    exp: InlineArray[ComplexFloat32,samples],
    out result: InlineArray[ComplexFloat32,samples]
):
    result = InlineArray[ComplexFloat32,samples](uninitialized=True)
    @parameter
    for s in range(samples):
        result[s] = err[err_fn](obs[s], exp[s])
