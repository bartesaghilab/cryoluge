
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
    var _res_factor: Ang[dtype]

    fn __init__[dim: Dimension](
        out self,
        shells: FourierShells[dim],
        pixel_size: Ang[dtype]
    ):
        self._values = List[Scalar[dtype]](length=shells.shelli_max(), fill=Scalar[dtype](0))
        self._res_factor = Ang[dtype](shells.count_at_unity)*pixel_size
    
    fn resolution(self, shelli: Int) -> Ang[dtype]:
        return self._res_factor/Scalar[dtype](shelli)

    fn read_from_statistics(
        mut self,
        stats: ResolutionStatistics[dtype]
    ) raises:
        var shelli = 0

        # the zero shell is always 1000 for some reason
        self._values[shelli] = 1000
        shelli += 1
        
        # read as many shells from the stats as we can
        for record in stats.records:

            # make sure the statistics match our shell configuration
            if Scalar[dtype](shelli + 1) != record.shell:
                raise Error(
                    "Expected to read shell ", (shelli + 1),
                    ", but found shell ", record.shell, " instead"
                )
            var res = self.resolution(shelli)
            if (record.resolution - res).abs() > 1e-3:
                raise Error(
                    "Expected to read shell ", (shelli + 1), " with resolution ", res,
                    ", but found resolution ", record.resolution, " instead"
                )

            # all is well: read the particle SSNR
            self._values[shelli] = record.particle_ssnr
            shelli += 1

        # fill the rest with 0
        while shelli < len(self._values):
            self._values[shelli] = 0
            shelli += 1

    fn generate(
        mut self,
        *,
        mass_kda: KDa[dtype]
    ):
        # model a spherical particle of the given mass
        var diameter_a = SphericalParticle(mass_kda=mass_kda).diameter_a().value
        
        # pre-calulate some values
        var precalc_1 = ( mass_kda.value**1.5 )/2200

        # the zero shell is always 1000 for some reason
        self._values[0] = 1000
        
        for shelli in range(1, len(self._values)):
            # Approximate formula derived from part_SSNR curve for VSV-L
            var resolution = self.resolution(shelli).value
            self._values[shelli] = precalc_1*(800*exp(-3.5*diameter_a/resolution) + exp(-25.0/resolution))

    fn __getitem__(self, shelli: Int) -> Scalar[dtype]:
        return self._values[shelli]
