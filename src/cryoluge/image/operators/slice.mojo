
from cryoluge.math import Dimension, Vec, Matrix
from cryoluge.fft import FFTImage


struct SliceOperator[
    dtype: DType,
    src_origin: Origin[mut=False]
]:
    comptime Src = FFTImage[Dimension.D3,dtype]
    comptime Dst = FFTImage[Dimension.D2,dtype]

    var _src: Pointer[Self.Src, src_origin]
    var _res_limit2: Float32

    fn __init__(
        out self,
        ref [src_origin] src: Self.Src,
        res_limit: Float32
    ):
        self._src = Pointer(to=src)

        var x_size = Float32(src.coords().sizes_real().x())
        self._res_limit2 = (res_limit*x_size)**2

    fn eval(
        self,
        mut dst: Self.Dst,
        rot: Matrix.D3[DType.float32],
        *,
        f: Vec[Int,Self.Dst.dim],
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0),
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[dtype]
    ):
        ref src = self._src[]

        if f == Vec.D2[Int](x=0, y=0):
            pixel = origin_value
        else:

            var f_f = f.map_float32()

            if f_f.len2() <= self._res_limit2:

                # rotate the sample point into 3d
                var freq_3d_f = rot*f_f.lift(z=0)

                # do the linear interpolation
                var v = src.get(f_lerp=freq_3d_f).or_else(out_of_range)

                # save to the output
                if dst.coords().needs_conjugation(f=f):
                    v = v.conj()
                pixel = v
            else:
                pixel = out_of_range

    fn eval(
        self,
        mut dst: Self.Dst,
        rot: Matrix.D3[DType.float32],
        *,
        i: Vec[Int,Self.Dst.dim],
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0),
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[dtype]
    ):
        var f = dst.coords().i2f(i)
        pixel = self.eval(dst, rot, f=f, out_of_range=out_of_range, origin_value=origin_value)

    fn apply(
        self,
        *,
        mut to: FFTImage.D2[dtype],
        rot: Matrix.D3[DType.float32],
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0),
        origin_value: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ):
        @parameter
        fn func(i: to.Vec[Int]):
            to.complex[i] = self.eval(to, rot, i=i, out_of_range=out_of_range, origin_value=origin_value)

        to.complex.iterate[func]()
