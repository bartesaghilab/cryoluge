
from cryoluge.math import Dimension, Vec, Matrix
from cryoluge.fft import FFTImage


struct SliceOperator[dtype: DType](
    Copyable,
    Movable
):

    comptime Src = FFTImage[Dimension.D3,dtype]
    comptime Dst = FFTImage[Dimension.D2,dtype]

    var _dst_sizes_real: Vec.D2[Int]
    var _res_limit2: Float32

    fn __init__(
        out self,
        *,
        src_sizes_real: Vec.D3[Int],
        dst_sizes_real: Vec.D2[Int],
        res_limit: Float32
    ):
        self._dst_sizes_real = dst_sizes_real.copy()

        var x_size = Float32(src_sizes_real.x())
        self._res_limit2 = (res_limit*x_size)**2

    fn eval[
        *,
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        rot: Matrix.D3[DType.float32],
        f: Vec[Int,Self.Dst.dim],
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[dtype]
    ):
        if f == Vec.D2[Int](x=0, y=0):
            pixel = origin_value
        else:

            var f_f = f.map_float32()

            if f_f.len2() <= self._res_limit2:

                # rotate the sample point into 3d
                var freqs = rot*f_f.lift(z=0)

                # do the linear interpolation
                var v = src.get[or_else=out_of_range](f_lerp=freqs)

                # save to the output
                pixel = v
            else:
                pixel = out_of_range
    
    fn eval[
        *,
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        rot: Matrix.D3[DType.float32],
        i: Vec[Int,Self.Dst.dim],
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[dtype]
    ):
        var f = FFTCoords(self._dst_sizes_real).i2f(i)
        pixel = self.eval[out_of_range=out_of_range](
            src=src,
            rot=rot,
            f=f,
            origin_value=origin_value
        )

    fn apply[
        *,
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        mut to: FFTImage.D2[dtype],
        rot: Matrix.D3[DType.float32],
        origin_value: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ):
        @parameter
        fn func(i: to.Vec[Int]):
            to.complex[i] = self.eval[out_of_range=out_of_range](
                src=src,
                rot=rot,
                i=i,
                origin_value=origin_value
            )

        to.complex.iterate[func]()
