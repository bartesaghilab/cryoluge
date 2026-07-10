
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

    var _values: List[Scalar[dtype]]
    var _num_shells_at_unity: Int

    fn __init__[dim: Dimension](
        out self,
        shells: FourierShells[dim]
    ):
        self._values = List[Scalar[dtype]](length=shells.shelli_max(), fill=Scalar[dtype](0))
        self._num_shells_at_unity = shells.count_at_unity

    fn generate(
        mut self,
        *,
        mass_kda: KDa[dtype],
        pixel_size: Ang[dtype]
    ):
        # model a spherical particle of the given mass
        var diameter_a = SphericalParticle(mass_kda=mass_kda).diameter_a().value
        
        # pre-calulate some values
        var precalc_1 = ( mass_kda.value**1.5 )/2200
        var precalc_2 = Scalar[dtype](self._num_shells_at_unity)*pixel_size.value

        # the zero shell is always 1000 for some reason
        self._values[0] = 1000
        
        for shelli in range(1, len(self._values)):
            # Approximate formula derived from part_SSNR curve for VSV-L
            var resolution = precalc_2/Scalar[dtype](shelli)
            self._values[shelli] = precalc_1*(800*exp(-3.5*diameter_a/resolution) + exp(-25.0/resolution))

    fn __getitem__(self, shelli: Int) -> Scalar[dtype]:
        return self._values[shelli]
