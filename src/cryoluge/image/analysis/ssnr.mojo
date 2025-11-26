
from math import exp

from cryoluge.math.units import KDa, Ang, pi


struct SSNR[dtype: DType](
    Copyable,
    Movable
):
    """
    Spectral Signal-to-Noise Ratio
    https://en.wikipedia.org/wiki/Spectral_signal-to-noise_ratio
    See the wiki above for more info.
    """

    var pixel_size: Scalar[dtype]
    var particle_diameter_a: Ang[dtype]
    var num_bins: Int
    var _precalc_1: Scalar[dtype]

    fn __init__(
        out self,
        *,
        pixel_size: Scalar[dtype],
        mass_kda: KDa[dtype],
        num_bins: Int
    ):
        self.pixel_size = pixel_size
        self.num_bins = num_bins
        
        # calculate the particle diameter
        var base = 3*mass_kda.to_ang3()/4/pi[dtype].value
        var exp = Scalar[dtype](1.0/3.0)
        self.particle_diameter_a = 2*( base**exp )

        # pre-calulate some values
        self._precalc_1 = ( mass_kda.value**1.5 )/2200
    
    fn __getitem__(self, bin: Int, out v: Scalar[dtype]):
        if bin == 0:
            v = 1000
        else:
            # Approximate formula derived from part_SSNR curve for VSV-L
            var resolution = Scalar[dtype](self.num_bins)*self.pixel_size/Scalar[dtype](bin)
            v = self._precalc_1
                * (800*exp(-3.5*self.particle_diameter_a.value/resolution) + exp(-25.0/resolution))
