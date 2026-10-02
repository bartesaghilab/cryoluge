
@inline(.always)
def zero[dtype: DType, simd_width: Int = 1](out zero: ComplexSIMD[dtype,simd_width]):
    zero = ComplexSIMD[dtype,simd_width](0, 0)


@inline(.always)
def slice[
    dtype: DType,
    //,
    i: Int = 0,
    simd_width_out: Int = 1
](
    c: ComplexSIMD[dtype,_],
    out result: ComplexSIMD[dtype,simd_width_out]
):
    result = ComplexSIMD[dtype,simd_width_out](
        re=c.re.slice[simd_width_out,offset=i](),
        im=c.im.slice[simd_width_out,offset=i]()
    )


@inline(.always)
def _simd_slice[
    dtype: DType,
    //,
    width: Int = 1
](
    v_in: SIMD[dtype,_],
    i: Int,
    out v_out: SIMD[dtype,width]
):
    v_out = 0
    @parameter
    for w in range(width):
        v_out[w] = v_in[i+w]


@inline(.always)
def slice[
    dtype: DType,
    //,
    simd_width_out: Int = 1
](
    c: ComplexSIMD[dtype,_],
    i: Int,
    out result: ComplexSIMD[dtype,simd_width_out]
):
    result = ComplexSIMD[dtype,simd_width_out](
        re=_simd_slice[simd_width_out](c.re, i),
        im=_simd_slice[simd_width_out](c.im, i)
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
    @parameter
    for n in range(len):
        c.re[i+n] = src.re[i_src+n]
        c.im[i+n] = src.im[i_src+n]


@inline(.always)
def pack[
    dtype: DType,
    simd_width_src: Int,
    //,
    simd_width: Int,
](
    *cs: ComplexSIMD[dtype,simd_width_src],
    out result: ComplexSIMD[dtype,simd_width]
):
    debug_assert(
        len(cs)*simd_width_src == simd_width,
        "Expected ", simd_width, " scalar elements, but got ", len(cs)*simd_width_src, " instead"
    )
    result = zero[dtype,simd_width]()
    for i in range(len(cs)):
        splice[simd_width_src](result, i*simd_width_src, cs[i])
    # TODO: maybe future versions of mojo will let us run this loop at compile time
    #       then we can use SIMD.insert()
