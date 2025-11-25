
from math import pi, cos, sin, floor

from cryoluge.math import Dimension, Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage
from cryoluge.ctf import CTF


struct FFTImage[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: Vec[Int,dim]
    var complex: ComplexImage[dim,dtype]

    comptime D1 = FFTImage[Dimension.D1,_]
    comptime D2 = FFTImage[Dimension.D2,_]
    comptime D3 = FFTImage[Dimension.D3,_]
    comptime Vec = ComplexImage[dim,dtype].Vec
    comptime PixelType = ComplexImage[dim,dtype].PixelType
    comptime PixelVec = ComplexImage[dim,dtype].PixelVec
    comptime ScalarType = ComplexImage[dim,dtype].ScalarType
    comptime ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[dim,dtype](fft_coords.sizes_fourier(), alignment=alignment)

    fn __init__(out self, *, of: Image[dim,dtype], alignment: Optional[Int] = None):
        ref real = of
        self = Self(real.sizes(), alignment=alignment)

    fn coords(self) -> FFTCoords[dim, origin=origin_of(self.sizes_real)]:
        return FFTCoords(self.sizes_real)

    fn crop(self, *, mut to: Self):
        ref dst = to

        # make sure the destination image is smaller (or the same size) as this one
        @parameter
        for d in range(dim.rank):
            debug_assert(
                dst.sizes_real[d] <= self.sizes_real[d],
                "Crop destination real sizes ", dst.sizes_real,
                " must be smaller (or same size) as this source real sizes ", self.sizes_real
            )

        # sample into the dst image
        @parameter
        fn sample(i: Self.Vec[Int]):
            dst.complex[i] = self.complex[self.coords().f2i(dst.coords().i2f(i))]

        dst.complex.iterate[sample]()

    # TODO: separate from iterate()
    fn phase_shift(mut self, shift: Self.Vec[Scalar[dtype]]):
        
        @parameter
        fn func(i: Self.Vec[Int]):
            var freq = self.coords().i2f(i).map_scalar[dtype]()
            var sizes_real = self.coords().sizes_real().map_scalar[dtype]()
            var phase = 0 - (freq*shift*2*pi/sizes_real).sum()
            self.complex[i=i] *= ComplexScalar[dtype](re=cos(phase), im=sin(phase))

        self.complex.iterate[func]()

    fn get(
        self,
        *,
        f: Self.Vec[Int],
        out v: Optional[ComplexScalar[dtype]]
    ):
        var conj = self.coords().needs_conjugation(f=f)
        var i = self.coords().maybe_f2i(f, needs_conj=conj)

        if i is None:
            v = None
            return

        v = self.complex.get(i.value())
        if v is not None and conj:
            v = v.value().conj()

    fn get(
        self: FFTImage[dim,dtype],
        *,
        f_lerp: Self.Vec[Float32]
    ) -> Optional[ComplexScalar[dtype]]:
        ref f = f_lerp

        # discretize the frequency coordinates
        @parameter
        fn func_start(v: Float32) -> Int:
            return Int(floor(v))
        var start = f.map[mapper=func_start]()

        # build the multi-dimensional delta vectors (at compile-time)
        fn build_deltas(out deltas: List[Vec[Int,dim]]):
            deltas = [
                Vec[Int,dim](fill=0)
            ]
            for d in range(dim.rank):
                for i in range(len(deltas)):
                    var delta = deltas[i].copy()
                    delta[d] = 1
                    deltas.append(delta^)

        sum = ComplexScalar[dtype](0, 0)

        @parameter
        for delta in build_deltas():
            var f_sample = start + materialize[delta]()

            # TEMP: technically, the sample might need conjugation,
            #       and the coordinate inversion might put it back in-range,
            #       but csp1 doesn't do that,
            #       so add another explicit (and unecessary) range check here to match csp1 behavior
            if not self.coords().f_in_range(f_sample):
                return None
            
            var v = self.get(f=f_sample)
            if v is None:
                return None
            var dists = (f - f_sample.map_float32()).abs()
            var weight = Scalar[dtype]((1 - dists).product())
            sum = sum + v.value()*weight

        return sum


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


struct WhitenOperator[
    dim: Dimension,
    dtype: DType,
    sum_dtype: DType=dtype
]:
    var shells: FourierShells[dim]
    var res_limit: Int
    var _sum: List[Scalar[sum_dtype]]
    var _count: List[Int]

    fn __init__(
        out self,
        shells: FourierShells[dim],
        *,
        res_limit: Optional[Int] = None
    ):
        self.shells = shells.copy()
        self.res_limit = res_limit.or_else(shells.count)
        self._sum = List[Scalar[sum_dtype]](length=shells.count, fill=0)
        self._count = List[Int](length=shells.count, fill=0)

    fn reset(mut self):
        for shelli in range(self.shells.count):
            self._sum[shelli] = 0
            self._count[shelli] = 0

    fn collect_statistics(mut self, *, f: Vec[Int,dim], v: ComplexScalar[dtype]):
        var dist2 = v.squared_norm()
        if dist2 > 0:
            # TODO: NEXTTIME: the counts (and sums) are wrong! find out why! (is shelli off? is freq2 off?)
            var shelli = self.shells.shelli[dtype](f=f)
            if shelli <= self.res_limit:
                self._sum[shelli] += Scalar[sum_dtype](dist2)
                self._count[shelli] += 1

    fn normalize(mut self):
        for shelli in range(self.shells.count):
            if self._count[shelli] > 0:
                self._sum[shelli] = sqrt(self._sum[shelli]/Scalar[sum_dtype](self._count[shelli]))

    fn eval(self, *, f: Vec[Int,dim], v: ComplexScalar[dtype], out result: ComplexScalar[dtype]):
        var shelli = self.shells.shelli[dtype](f=f)
        if shelli <= self.res_limit and self._count[shelli] > 0:
            var sum = Scalar[dtype](self._sum[shelli])
            result = ComplexScalar[dtype](re=v.re/sum, im=v.im/sum)
        else:
            result = ComplexScalar[dtype](0, 0)

    fn apply(mut self, mut image: FFTImage[dim,dtype]):

        self.reset()

        @parameter
        fn func1(i: Vec[Int,dim]):
            var v = image.complex[i=i]
            var f = image.coords().i2f(i)
            self.collect_statistics(f=f, v=v)

        image.complex.iterate[func1]()

        self.normalize()

        @parameter
        fn func2(i: Vec[Int,dim]):
            var v = image.complex[i=i]
            var f = image.coords().i2f(i)
            image.complex[i=i] = self.eval(f=f, v=v)

        image.complex.iterate[func2]()


struct WeightBySSNROperator[
    dtype: DType,
    ssnr_origin: Origin[mut=False]
]:
    var sizes_real: Vec.D2[Int]
    var ctf: CTF[dtype]
    var shells: FourierShells[Dimension.D2]
    var ssnr: Pointer[SSNR[dtype], ssnr_origin]
    var _scale_factor: Scalar[dtype]

    fn __init__(
        out self,
        *,
        sizes_real: Vec.D2[Int],
        ctf: CTF[dtype],
        shells: FourierShells[Dimension.D2],
        ref [ssnr_origin] ssnr: SSNR[dtype]
    ):
        self.sizes_real = sizes_real.copy()
        self.ctf = ctf.copy()
        self.shells = shells.copy()
        self.ssnr = Pointer(to=ssnr)

        # compute the scale factor        
        var particle_diameter2_px = ssnr.particle_diameter_a.to_px(ctf.pixel_size)**2
        var particle_area2_px = pi*particle_diameter2_px/4
        self._scale_factor = particle_area2_px.value/sizes_real.x()/sizes_real.y()

    fn eval(
        self,
        *,
        f: Vec.D2[Int],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        ref ssnr = self.ssnr[]

        var ctf2 = self.ctf.eval(f=f, sizes_real=self.sizes_real)**2
        var freq2 = FFTCoords(self.sizes_real).freqs[dtype](f=f).len2()
        var shelli = self.shells.shelli[dtype](f=f)

        result = v

        # TODO: magic number?
        if freq2 <= 0.25:
            result *= sqrt(1 + ctf2*abs(ssnr[shelli])*self._scale_factor)

    fn apply(self, mut image: FFTImage[Dimension.D2,dtype]):
        
        @parameter
        fn func(i: Vec.D2[Int]):
            var v = image.complex[i=i]
            var f = image.coords().i2f(i)
            image.complex[i=i] = self.eval(f=f, v=v)

        image.complex.iterate[func]()
