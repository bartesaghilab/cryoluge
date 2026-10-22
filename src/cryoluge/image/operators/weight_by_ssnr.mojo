
from std.math import sqrt
from std.complex import ComplexScalar

from cryoluge.math import Vec
from cryoluge.math.units import pi, Ang, KDa
from cryoluge.ctf import CTF
from cryoluge.image.analysis import FourierShells, SSNR
from cryoluge.fft import FFTImage
from cryoluge.model import SphericalParticle


struct WeightBySSNROperator[
    dtype: DType,
    shells_origin: Origin[mut=False]
] where dtype.is_floating_point():

    var shells: Pointer[FourierShells[2],Self.shells_origin]
    var _factors: List[Scalar[Self.dtype]]

    def __init__(
        out self,
        *,
        mass_kda: KDa[Scalar[Self.dtype]],
        pixel_size: Ang[Scalar[Self.dtype]],
        sizes_real: Vec[2,.int],
        ref [Self.shells_origin] shells: FourierShells[2],
        ssnr: SSNR[Self.dtype]
    ):
        self.shells = Pointer(to=shells)

        # check the SSNR size
        debug_assert(
            len(ssnr) == len(shells),
            "SSNR entry count (", len(ssnr), ") doesn't match fourier shell count (", len(shells), ")"
        )

        # compute the scale factor (ratio of particle area to image area)
        var particle = SphericalParticle(mass_kda=mass_kda)
        var particle_area_px = particle.cross_area_a().to_px(pixel_size**2)
        var scale_factor = particle_area_px.value
            / Scalar[Self.dtype](sizes_real.x())
            / Scalar[Self.dtype](sizes_real.y())

        # pre-compute the shell factors
        self._factors = List[Scalar[Self.dtype]](capacity=len(shells))
        for shelli in range(len(shells)):
            var factor = abs(ssnr[shelli])*scale_factor
            self._factors.append(factor)

    def eval(
        self,
        *,
        freq_norm: Scalar[Self.dtype],
        ctf2: Scalar[Self.dtype],
        v: ComplexScalar[Self.dtype],
        out result: ComplexScalar[Self.dtype]
    ):
        result = v

        # make sure the normalized frequency is in range [-0.5,0.5]
        # ie, only apply to the inscribed circle of the square image
        if freq_norm**2 <= 0.25:
            var shelli = self.shells[].shelli(freq_norm=freq_norm)
            result *= sqrt(1 + ctf2*self._factors[shelli])

    def apply(self, ctf: CTF[Self.dtype], mut image: FFTImage[2,Self.dtype]):

        var coords = image.coords()
        
        def func(i: Vec[2,.int]) {self, mut image, coords, ctf}:
            var v = image.complex[i=i]
            var f = coords.i2f(i)
            var f_norm = coords.f_norm[Self.dtype](f=f)
            var ctfi2 = ctf.eval(f_norm=f_norm)**2
            image.complex[i=i] = self.eval(freq_norm=f_norm.len(), ctf2=ctfi2, v=v)

        image.complex.pixels().iterate(func)
