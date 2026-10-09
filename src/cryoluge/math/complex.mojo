
from std.complex import ComplexSIMD


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
