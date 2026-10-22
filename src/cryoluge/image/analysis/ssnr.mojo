
from std.math import exp

from cryoluge.math.units import KDa, Ang, pi
from cryoluge.model import SphericalParticle
from cryoluge.image.analysis.shells import FourierShells
from cryoluge.image.analysis.resolution_statistics import ResolutionStatistics


struct SSNR[dtype: DType](
    Copyable,
    Movable,
    Sized
):
    """
    Spectral Signal-to-Noise Ratio
    https://en.wikipedia.org/wiki/Spectral_signal-to-noise_ratio
    See the wiki above for more info.
    """

    var _values: List[Scalar[Self.dtype]]
    var _res_factor: Ang[Scalar[Self.dtype]]

    def __init__[dim: Int](
        out self,
        shells: FourierShells[dim],
        pixel_size: Ang[Scalar[Self.dtype]]
    ):
        self._values = List[Scalar[Self.dtype]](length=len(shells), fill=Scalar[Self.dtype](0))
        self._res_factor = pixel_size*Scalar[Self.dtype](shells.count_at_unity)
    
    def resolution(self, shelli: Int) -> Ang[Scalar[Self.dtype]]:
        return self._res_factor/Scalar[Self.dtype](shelli)

    def read_from_statistics(
        mut self,
        stats: ResolutionStatistics[Self.dtype]
    ) raises:
        var shelli = 0

        # the zero shell is always 1000 for some reason
        self._values[shelli] = 1000
        shelli += 1
        
        # read as many shells from the stats as we can
        for record in stats.records:

            # make sure the statistics match our shell configuration
            if Scalar[Self.dtype](shelli + 1) != record.shell:
                raise Error(
                    "Expected to read shell ", (shelli + 1),
                    ", but found shell ", record.shell, " instead"
                )
            var res = self.resolution(shelli)
            if (record.resolution - res).abs() > Scalar[Self.dtype](1e-3):
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

    def generate(
        mut self,
        *,
        mass_kda: KDa[Scalar[Self.dtype]]
    ) where Self.dtype.is_floating_point():

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

    def __len__(self) -> Int:
        return len(self._values)

    def __getitem__(self, shelli: Int) -> Scalar[Self.dtype]:
        return self._values[shelli]
