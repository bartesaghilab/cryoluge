
from std.complex import ComplexScalar

from cryoluge.math import Vec, Matrix
from cryoluge.fft import FFTImage, FFTCoords


struct SliceOperator[dtype: DType](
    Copyable,
    Movable
):

    comptime Src = FFTImage[3,Self.dtype]
    comptime Dst = FFTImage[2,Self.dtype]

    var _dst_sizes_real: Vec[2,.int]
    var _res_limit2: Float32

    def __init__(
        out self,
        *,
        src_sizes_real: Vec[3,.int],
        dst_sizes_real: Vec[2,.int],
        res_limit: Float32
    ):
        self._dst_sizes_real = dst_sizes_real.copy()

        var x_size = Float32(src_sizes_real.x())
        self._res_limit2 = (res_limit*x_size)**2

    def eval[
        *,
        out_of_range: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        rot: Matrix[3,3,DType.float32],
        f: Vec[Self.Dst.dim,.int],
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[Self.dtype]
    ):
        if f == Vec[2,.int](x=0, y=0):
            pixel = origin_value
        else:

            var f_f = f.map_dtype[.float32]()

            if f_f.len2() <= self._res_limit2:

                # rotate the sample point into 3d
                var freqs = rot*f_f.lift(z=0)

                # do the linear interpolation
                var v = src.get[or_else=out_of_range](f_lerp=freqs)

                # save to the output
                pixel = v
            else:
                pixel = out_of_range
    
    def eval[
        *,
        out_of_range: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        rot: Matrix[3,3,DType.float32],
        i: Vec[Self.Dst.dim,.int],
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0),
        out pixel: ComplexScalar[Self.dtype]
    ):
        var f = FFTCoords(self._dst_sizes_real).i2f(i)
        pixel = self.eval[out_of_range=out_of_range](
            src=src,
            rot=rot,
            f=f,
            origin_value=origin_value
        )

    def apply[
        *,
        out_of_range: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)
    ](
        self,
        *,
        src: Self.Src,
        mut to: FFTImage[2,Self.dtype],
        rot: Matrix[3,3,.float32],
        origin_value: ComplexScalar[Self.dtype] = ComplexScalar[Self.dtype](0, 0)
    ):
        def func(i: to.Vec[.int]) {self, mut to, src, rot, origin_value}:
            to.complex[i] = self.eval[out_of_range=out_of_range](
                src=src,
                rot=rot,
                i=i,
                origin_value=origin_value
            )

        to.complex.pixels().iterate(func)
