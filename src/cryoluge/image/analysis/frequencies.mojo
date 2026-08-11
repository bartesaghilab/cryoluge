
from utils.numerics import inf

from cryoluge.math import clamp, Vec
from cryoluge.math.units import Ang
from cryoluge.fft import FFTImage, FFTCoords
from cryoluge.image.analysis import FourierShells


@fieldwise_init
struct FrequencyLimits[dtype: DType](
    Copyable,
    Movable
):
    var freq_norm2_lo: Scalar[dtype]
    var freq_norm2_hi: Scalar[dtype]

    fn __init__(
        out self,
        *,
        res_limit_lo: Ang[dtype],
        res_limit_hi: Ang[dtype],
        pixel_size: Ang[dtype]
    ):

        self.freq_norm2_lo = Scalar[dtype](0)
        if res_limit_lo != 0:
            self.freq_norm2_lo = (pixel_size.value/res_limit_lo.value)**2

        self.freq_norm2_hi = Scalar[dtype](0)
        if res_limit_hi != 0:
            self.freq_norm2_hi = clamp(pixel_size.value/res_limit_hi.value, max=0.5)**2

    @staticmethod
    fn none(out self: Self):
        self = Self(
            freq_norm2_lo = Scalar[dtype](0),
            freq_norm2_hi = inf[dtype]()
        )

    @always_inline
    fn contains[
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        freq_norm2: Scalar[dtype],
        out contains: Bool
    ):
        contains = True

        @parameter
        if inclusive_lo:
            contains = contains and freq_norm2 >= self.freq_norm2_lo
        else:
            contains = contains and freq_norm2 > self.freq_norm2_lo

        @parameter
        if inclusive_hi:
            contains = contains and freq_norm2 <= self.freq_norm2_hi
        else:
            contains = contains and freq_norm2 < self.freq_norm2_hi

    fn shell_indices[dim: Int](self, shells: FourierShells[dim]) -> Tuple[Int,Int]:
        """Returns shell index lower,upper(exclusive)."""
        var shelli_min = shells.shelli(freq_norm2=self.freq_norm2_lo)
        var shelli_max = shells.shelli(freq_norm2=self.freq_norm2_hi)
        return (shelli_min, shelli_max + 1)

    fn intersect(
        mut self,
        *,
        freq_norm_lo: Optional[Scalar[dtype]] = None,
        freq_norm_hi: Optional[Scalar[dtype]] = None
    ):
        # bring up the lower boundary, if needed
        if freq_norm_lo is not None:
            self.freq_norm2_lo = max(self.freq_norm2_lo, freq_norm_lo.value()**2)

        # bring down the upper boundary, if needed
        if freq_norm_hi is not None:
            self.freq_norm2_hi = min(self.freq_norm2_hi, freq_norm_hi.value()**2)

    fn checker[dim: Int](self, sizes_real: Vec[dim,Int], out checker: FrequencyLimitsChecker[dim,dtype]):
        checker = FrequencyLimitsChecker(self, sizes_real)


struct FrequencyLimitsChecker[dim: Int, dtype: DType](
    Copyable,
    Movable
):
    var _limits: FrequencyLimits[dtype]
    var _sizes_norm2: Vec[dim,Scalar[dtype]]

    fn __init__(
        out self,
        limits: FrequencyLimits[dtype],
        sizes_real: Vec[dim,Int]
    ):
        self._limits = limits.copy()
        self._sizes_norm2 = FFTCoords(sizes_real).sizes_voxel_norm[dtype]()**2

    @always_inline
    fn contains[
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        f: Vec[dim,Scalar[dtype]],
        out contains: Bool
    ):
        var freq_norm2 = self._sizes_norm2.inner_product(f**2)
        contains = self._limits.contains[inclusive_lo=inclusive_lo, inclusive_hi=inclusive_hi](freq_norm2=freq_norm2)
