
from std.complex import ComplexSIMD, ComplexScalar


@inline(.always)
def zero[dtype: DType, slen: SIMDLength = 1](out zero: ComplexSIMD[dtype,slen]):
    zero = ComplexSIMD[dtype,slen](0, 0)


@inline(.always)
def slice[
    dtype: DType,
    //,
    i: Int = 0,
    slen_out: SIMDLength = 1
](
    c: ComplexSIMD[dtype,_],
    out result: ComplexSIMD[dtype,slen_out]
):
    result = ComplexSIMD[dtype,slen_out](
        re=c.re.slice[slen_out,offset=i](),
        im=c.im.slice[slen_out,offset=i]()
    )


@inline(.always)
def _simd_slice[
    dtype: DType,
    //,
    slen: SIMDLength = 1
](
    v_in: SIMD[dtype,_],
    i: Int,
    out v_out: SIMD[dtype,slen]
):
    v_out = 0
    comptime for w in range(slen):
        v_out[w] = v_in[i+w]


@inline(.always)
def slice[
    dtype: DType,
    //,
    slen_out: SIMDLength = 1
](
    c: ComplexSIMD[dtype,_],
    i: Int,
    out result: ComplexSIMD[dtype,slen_out]
):
    result = ComplexSIMD[dtype,slen_out](
        re=_simd_slice[slen_out](c.re, i),
        im=_simd_slice[slen_out](c.im, i)
    )


@inline(.always)
def splice[
    dtype: DType,
    //,
    len: Int = 1
](
    mut c: ComplexSIMD[dtype,_],
    i: Int,
    src: ComplexSIMD[dtype,_],
    i_src: Int = 0
):
    comptime for n in range(len):
        c.re[i+n] = src.re[i_src+n]
        c.im[i+n] = src.im[i_src+n]


@inline(.always)
def pack[
    dtype: DType,
    slen_src: SIMDLength,
    //,
    slen: SIMDLength,
](
    *cs: ComplexSIMD[dtype,slen_src],
    out result: ComplexSIMD[dtype,slen]
):
    var slen_obs = len(cs)*slen_src
    debug_assert(
        slen_obs == slen,
        "Expected ", Int(slen), " scalar elements, but got ", slen_obs, " instead"
    )

    result = zero[dtype,slen]()
    for i in range(len(cs)):
        splice[slen_src](result, i*slen_src, cs[i])
    # TODO: maybe future versions of mojo will let us run this loop at compile time
    #       then we can use SIMD.insert()


@inline(.always)
def splat[dtype: DType, //, slen: SIMDLength](v: ComplexSIMD[dtype,1]) -> ComplexSIMD[dtype,slen]:
    return ComplexSIMD[dtype,slen](
        re = SIMD[dtype,slen](v.re),
        im = SIMD[dtype,slen](v.im)
    )


@inline(.always)
def reduce_add[dtype: DType, slen: SIMDLength, //, dtype_out: DType = dtype](
    c: ComplexSIMD[dtype,slen],
    out result: ComplexScalar[dtype_out]
):
    result = ComplexScalar[dtype_out](
        re=SIMD[dtype_out,slen](c.re).reduce_add(),
        im=SIMD[dtype_out,slen](c.im).reduce_add()
    )


@inline(.always)
def elementwise_mul[dtype: DType, slen: SIMDLength](
    a: ComplexSIMD[dtype,slen],
    b: ComplexSIMD[dtype,slen],
    out result: ComplexSIMD[dtype,slen]
):
    result = ComplexSIMD[dtype,slen](
        re = a.re*b.re,
        im = a.im*b.im
    )


@inline(.always)
def mul[dtype: DType, slen: SIMDLength](
    c: ComplexSIMD[dtype,slen],
    s: SIMD[dtype,slen],
    out result: ComplexSIMD[dtype,slen]
):
    result = ComplexSIMD[dtype,slen](
        re = c.re*s,
        im = c.im*s
    )


@inline(.always)
def dot[dtype: DType, slen: SIMDLength, //, dtype_out: DType = dtype](
    a: ComplexSIMD[dtype,slen],
    b: ComplexSIMD[dtype,slen],
    out result: Scalar[dtype_out]
):
    var c = reduce_add[dtype_out](elementwise_mul(a, b))
    result = c.re + c.im


@inline(.always)
def select[dtype: DType, slen: SIMDLength](
    mask: SIMD[.bool,slen],
    true_val: ComplexSIMD[dtype,slen],
    false_val: ComplexSIMD[dtype,slen],
    out result: ComplexSIMD[dtype,slen]
):
    result = ComplexSIMD[dtype,slen](
        re = mask.select(true_val.re, false_val.re),
        im = mask.select(true_val.im, false_val.im)
    )
