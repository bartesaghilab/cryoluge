
@always_inline
fn zero[dtype: DType, simd_width: Int = 1](out zero: ComplexSIMD[dtype,simd_width]):
    zero = ComplexSIMD[dtype,simd_width](0, 0)


@always_inline
fn slice[
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


@always_inline
fn splice[
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


@always_inline
fn pack[
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
        "Expected ", simd_width, " scalar elements, but only got ", len(cs)*simd_width_src
    )
    result = zero[dtype,simd_width]()
    for i in range(len(cs)):
        splice[simd_width_src](result, i*simd_width_src, cs[i])
    # TODO: maybe future versions of mojo will let us run this loop at compile time
    #       then we can use SIMD.insert()
