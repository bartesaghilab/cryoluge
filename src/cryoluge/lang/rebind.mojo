

fn rebind_scalar[
    src_dtype: DType, //,
    dst_dtype: DType
](src: Scalar[src_dtype]) -> Scalar[dst_dtype]:
    # rebind() (apparently) allows implicit type conversions,
    # which can lead to super-hard-to-find errors,
    # so be explicit about type equality at compile-time
    constrained[
        src_dtype == dst_dtype,
        String("Argument has dtype of ", src_dtype, " which does not match expected dtype of ", dst_dtype)
    ]()
    return rebind[Scalar[dst_dtype]](src)
