
fn slice[
    dtype: DType,
    simd_width: Int = 1
](
    c: ComplexSIMD[dtype,simd_width],
    i: Int,
    out result: ComplexScalar[dtype]
):
    result = ComplexScalar[dtype](
        re=c.re[i],
        im=c.im[i]
    )


fn pack[
    dtype: DType,
    simd_width: Int
](
    *cs: ComplexScalar[dtype],
    out result: ComplexSIMD[dtype,simd_width]
):
    debug_assert(
        len(cs) == simd_width,
        "Expected ", simd_width, " elements, but only got ", len(cs)
    )
    result = ComplexSIMD[dtype,simd_width](0, 0)
    for i in range(len(cs)):
        result.re[i] = cs[i].re
        result.im[i] = cs[i].im
