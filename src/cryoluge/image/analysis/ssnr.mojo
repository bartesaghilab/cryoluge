
from math import exp

from cryoluge.math.units import KDa, Ang, pi
from cryoluge.model import SphericalParticle


struct SSNR[dtype: DType](
    Copyable,
    Movable
):
    """
    Spectral Signal-to-Noise Ratio
    https://en.wikipedia.org/wiki/Spectral_signal-to-noise_ratio
    See the wiki above for more info.
    """

    var spherical_particle: SphericalParticle[dtype]
    var _precalc_1: Scalar[dtype]
    var _precalc_2: Scalar[dtype]

    fn __init__[dim: Dimension](
        out self,
        *,
        pixel_size: Ang[dtype],
        mass_kda: KDa[dtype],
        shells: FourierShells[dim]
    ):
        # model a spherical particle of the given mass
        self.spherical_particle = SphericalParticle(mass_kda=mass_kda)
        
        # pre-calulate some values
        self._precalc_1 = ( mass_kda.value**1.5 )/2200
        self._precalc_2 = Scalar[dtype](shells.count_at_unity)*pixel_size.value
    
    fn __getitem__(self, bin: Int, out v: Scalar[dtype]):
        if bin == 0:
            v = 1000
        else:
            # Approximate formula derived from part_SSNR curve for VSV-L
            var resolution = self._precalc_2/Scalar[dtype](bin)
            v = self._precalc_1
                * (800*exp(-3.5*self.spherical_particle.diameter_a().value/resolution) + exp(-25.0/resolution))
