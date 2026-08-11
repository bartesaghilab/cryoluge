
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
    var freq2_lo: Scalar[dtype]
    var freq2_hi: Scalar[dtype]

    fn __init__(
        out self,
        *,
        res_limit_lo: Ang[dtype],
        res_limit_hi: Ang[dtype],
        pixel_size: Ang[dtype],
        freq2_hi_limits: Optional[Tuple[Scalar[dtype],Scalar[dtype]]] = None
    ):

        self.freq2_lo = Scalar[dtype](0)
        if res_limit_lo != 0:
            self.freq2_lo = (pixel_size.value/res_limit_lo.value)**2

        self.freq2_hi = Scalar[dtype](0)
        if res_limit_hi != 0:
            self.freq2_hi = (pixel_size.value/res_limit_hi.value)**2
            if freq2_hi_limits is not None:
                var (limit_min, limit_max) = freq2_hi_limits.value()
                self.freq2_hi = clamp(self.freq2_hi, min=limit_min, max=limit_max)

    @staticmethod
    fn none(out self: Self):
        self = Self(
            freq2_lo = Scalar[dtype](0),
            freq2_hi = inf[dtype]()
        )

    @always_inline
    fn contains[
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        freq2: Scalar[dtype],
        out contains: Bool
    ):
        contains = True

        @parameter
        if inclusive_lo:
            contains = contains and freq2 >= self.freq2_lo
        else:
            contains = contains and freq2 > self.freq2_lo

        @parameter
        if inclusive_hi:
            contains = contains and freq2 <= self.freq2_hi
        else:
            contains = contains and freq2 < self.freq2_hi

    fn shell_indices[dim: Int](self, shells: FourierShells[dim]) -> Tuple[Int,Int]:
        """Returns shell index lower,upper(exclusive)."""
        var shelli_min = shells.shelli(freq2=self.freq2_lo)
        var shelli_max = shells.shelli(freq2=self.freq2_hi)
        return (shelli_min, shelli_max + 1)

    fn intersect(
        mut self,
        *,
        freq_lo: Optional[Scalar[dtype]] = None,
        freq_hi: Optional[Scalar[dtype]] = None
    ):
        # bring up the lower boundary, if needed
        if freq_lo is not None:
            self.freq2_lo = max(self.freq2_lo, freq_lo.value()**2)

        # bring down the upper boundary, if needed
        if freq_hi is not None:
            self.freq2_hi = min(self.freq2_hi, freq_hi.value()**2)
