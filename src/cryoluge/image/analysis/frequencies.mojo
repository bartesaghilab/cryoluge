
from std.utils.numerics import inf

from cryoluge.math import clamp, Vec
from cryoluge.math.units import Ang
from cryoluge.fft import FFTImage, FFTCoords
from cryoluge.image.analysis.shells import FourierShells


@fieldwise_init
struct FrequencyLimits[dtype: DType](
    Copyable,
    Movable
):
    var freq_norm2_lo: Scalar[Self.dtype]
    var freq_norm2_hi: Scalar[Self.dtype]

    def __init__(
        out self,
        *,
        res_limit_lo: Ang[Scalar[Self.dtype]],
        res_limit_hi: Ang[Scalar[Self.dtype]],
        pixel_size: Ang[Scalar[Self.dtype]]
    ):

        self.freq_norm2_lo = Scalar[Self.dtype](0)
        if res_limit_lo != 0:
            self.freq_norm2_lo = (pixel_size.value/res_limit_lo.value)**2

        self.freq_norm2_hi = Scalar[Self.dtype](0)
        if res_limit_hi != 0:
            self.freq_norm2_hi = clamp(pixel_size.value/res_limit_hi.value, max=0.5)**2

    @staticmethod
    def none(out self: Self):
        self = Self(
            freq_norm2_lo = Scalar[Self.dtype](0),
            freq_norm2_hi = inf[Self.dtype]()
        )

    @inline(.always)
    def contains[
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        freq_norm2: Scalar[Self.dtype],
        out contains: Bool
    ):
        contains = True

        comptime if inclusive_lo:
            contains = contains and freq_norm2 >= self.freq_norm2_lo
        else:
            contains = contains and freq_norm2 > self.freq_norm2_lo

        comptime if inclusive_hi:
            contains = contains and freq_norm2 <= self.freq_norm2_hi
        else:
            contains = contains and freq_norm2 < self.freq_norm2_hi

    @inline(.always)
    def contains[
        slen: SIMDLength,
        //,
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        freq_norm2: SIMD[Self.dtype,slen],
        out contains: SIMD[DType.bool,slen]
    ):
        contains = SIMD[DType.bool,slen](fill=True)

        var freq_norm2_lo = SIMD[Self.dtype,slen](self.freq_norm2_lo)
        var freq_norm2_hi = SIMD[Self.dtype,slen](self.freq_norm2_hi)

        comptime if inclusive_lo:
            contains = contains & freq_norm2.ge(freq_norm2_lo)
        else:
            contains = contains & freq_norm2.gt(freq_norm2_lo)

        comptime if inclusive_hi:
            contains = contains & freq_norm2.le(freq_norm2_hi)
        else:
            contains = contains & freq_norm2.lt(freq_norm2_hi)

    def shell_indices[dim: Int](self, shells: FourierShells[dim]) -> Tuple[Int,Int]:
        """Returns shell index lower,upper(exclusive)."""
        var shelli_min = shells.shelli(freq_norm2=self.freq_norm2_lo)
        var shelli_max = shells.shelli(freq_norm2=self.freq_norm2_hi)
        return (shelli_min, shelli_max + 1)

    def intersect(
        mut self,
        *,
        freq_norm_lo: Optional[Scalar[Self.dtype]] = None,
        freq_norm_hi: Optional[Scalar[Self.dtype]] = None
    ):
        # bring up the lower boundary, if needed
        if freq_norm_lo is not None:
            self.freq_norm2_lo = max(self.freq_norm2_lo, freq_norm_lo.value()**2)

        # bring down the upper boundary, if needed
        if freq_norm_hi is not None:
            self.freq_norm2_hi = min(self.freq_norm2_hi, freq_norm_hi.value()**2)

    def checker[dim: Int](self, sizes_real: Vec[dim,.int], out checker: FrequencyLimitsChecker[dim,Self.dtype]):
        checker = FrequencyLimitsChecker(self, sizes_real)


struct FrequencyLimitsChecker[dim: Int, dtype: DType](
    Copyable,
    Movable
):
    var _limits: FrequencyLimits[Self.dtype]
    var _sizes_norm2: Vec[Self.dim,Self.dtype]

    def __init__(
        out self,
        limits: FrequencyLimits[Self.dtype],
        sizes_real: Vec[Self.dim,.int]
    ):
        self._limits = limits.copy()
        self._sizes_norm2 = FFTCoords(sizes_real).sizes_voxel_norm[Self.dtype]()**2

    @inline(.always)
    def contains[
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        f: Vec[Self.dim,Self.dtype],
        out contains: Bool
    ):
        var freq_norm2 = self._sizes_norm2.inner_product(f**2)
        contains = self._limits.contains[inclusive_lo=inclusive_lo, inclusive_hi=inclusive_hi](freq_norm2=freq_norm2)

    @inline(.always)
    def contains[
        slen: SIMDLength,
        *,
        inclusive_lo: Bool = True,
        inclusive_hi: Bool = False,
    ](
        self,
        *,
        f: Vec[Self.dim,Self.dtype,slen],
        out contains: SIMD[DType.bool,slen]
    ):
        var freq_norm2 = self._sizes_norm2.splat[slen]().inner_product(f**2)
        contains = self._limits.contains[inclusive_lo=inclusive_lo, inclusive_hi=inclusive_hi](freq_norm2=freq_norm2)
