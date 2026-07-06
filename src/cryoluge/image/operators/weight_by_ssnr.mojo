
from cryoluge.math import Dimension
from cryoluge.math.units import pi, Ang
from cryoluge.ctf import CTF
from cryoluge.image.analysis import FourierShells, SSNR
from cryoluge.fft import FFTImage


struct WeightBySSNROperator[
    dtype: DType,
    shells_origin: Origin[mut=False]
]:
    var shells: Pointer[FourierShells[Dimension.D2],shells_origin]
    var _factors: List[Scalar[dtype]]

    fn __init__(
        out self,
        *,
        pixel_size: Ang[dtype],
        sizes_real: Vec.D2[Int],
        ref [shells_origin] shells: FourierShells[Dimension.D2],
        ssnr: SSNR[dtype]
    ):
        self.shells = Pointer(to=shells)

        # compute the scale factor        
        var particle_diameter2_px = ssnr.particle_diameter_a.to_px(pixel_size)**2
        var particle_area2_px = particle_diameter2_px*pi[dtype].value/4
        var scale_factor = particle_area2_px.value/sizes_real.x()/sizes_real.y()

        # pre-compute the shell factors
        self._factors = List[Scalar[dtype]](capacity=len(shells))
        for shelli in range(len(shells)):
            var factor = abs(ssnr[shelli])*scale_factor
            self._factors.append(factor)

    fn eval(
        self,
        *,
        freq: Scalar[dtype],
        ctf2: Scalar[dtype],
        v: ComplexScalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        result = v

        # TODO: what is this magic number?
        if freq**2 <= 0.25:
            var shelli = self.shells[].shelli(freq=freq)
            result *= sqrt(1 + ctf2*self._factors[shelli])

    fn apply(self, ctf: CTF[dtype], mut image: FFTImage[Dimension.D2,dtype]):

        var coords = image.coords()
        
        @parameter
        fn func(i: Vec.D2[Int]):
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var freqs = coords.freqs[dtype](f=f)
            var freq = sqrt(freqs.len2())
            var ctfi2 = ctf.eval(freqs=freqs)**2
            image.complex[i=i] = self.eval(freq=freq, ctf2=ctfi2, v=v)

        image.complex.iterate[func]()

        # TEMP: extend lifetimes to work around compiler bug
        _ = coords