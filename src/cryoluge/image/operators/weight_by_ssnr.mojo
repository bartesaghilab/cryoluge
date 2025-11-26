
from cryoluge.math import Dimension
from cryoluge.math.units import pi
from cryoluge.ctf import CTF
from cryoluge.image import ComplexImage
from cryoluge.image.analysis import FourierShells, SSNR
from cryoluge.fft import FFTCoords


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
        var particle_area2_px = particle_diameter2_px*pi[dtype].value/4
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

    fn apply(self, mut image: ComplexImage[Dimension.D2,dtype]):
        
        @parameter
        fn func(i: Vec.D2[Int]):
            var v = image[i=i]
            var f = FFTCoords(self.sizes_real).i2f(i)
            image[i=i] = self.eval(f=f, v=v)

        image.iterate[func]()
