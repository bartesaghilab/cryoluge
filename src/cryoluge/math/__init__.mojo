
from std.math import sin, cos, pi as pi_std, min, max
from std.complex import ComplexScalar, ComplexSIMD

from . import error
from . import dimension
from .euler import *
from .vec import *
from .matrix import *
from .rand import *
from .stats import *
from .sphere import *
from .box import *


comptime pi[dtype: DType, slen: SIMDLength = 1]
    where dtype.is_floating_point()
    = Rad[SIMD[dtype,slen]](pi_std)


def is_even(n: Int) -> Bool:
    return (n & 1) == 0


def is_odd(n: Int) -> Bool:
    return not is_even(n)


def round[digits: Int, dtype: DType, slen: SIMDLength](v: SIMD[dtype,slen]) -> SIMD[dtype,slen]:
    # the SIMD.__round__(digits) implementation is slow
    # this is a lot faster, when we know the number of digits at compile-time
    comptime if digits == 0:
        return v
    else:
        comptime f = SIMD[dtype,slen]( 10**digits )
        comptime oof = 1.0/f
        return (v*f).__round__()*oof

    # TODO: would a bitwise right shift then left shift accomplish something similar?


def clamp[dtype: DType, slen: SIMDLength](
    n: SIMD[dtype,slen],
    out r: SIMD[dtype,slen],
    *,
    min: SIMD[dtype,slen]
):
    r = std.math.max(n, min)


def clamp[dtype: DType, slen: SIMDLength](
    n: SIMD[dtype,slen],
    out r: SIMD[dtype,slen],
    *,
    max: SIMD[dtype,slen]
):
    r = std.math.min(n, max)


def clamp[dtype: DType, slen: SIMDLength](
    n: SIMD[dtype,slen],
    out r: SIMD[dtype,slen],
    *,
    min: SIMD[dtype,slen],
    max: SIMD[dtype,slen]
):
    r = std.math.min(std.math.max(n, min), max)


def ladder[slen: SIMDLength](out v: SIMD[DType.int,slen]):
    """
    Returns a vector containing all the integers in the range [0,slen) in order.
    """
    v = SIMD[DType.int,slen]()
    comptime for i in range(slen):
        v[i] = i


def ease_linear[dtype: DType, slen: SIMDLength](v_in: SIMD[dtype,slen], out v_out: SIMD[dtype,slen]):
    """
    Easing function over [0,1] with a linear shape, essentially a no-op.
    """
    v_out = v_in


def ease_cos[dtype: DType, slen: SIMDLength](v_in: SIMD[dtype,slen], out v_out: SIMD[dtype,slen])
    where dtype.is_floating_point():
    """
    Easing function over [0,1] with a cos shape.
    This curve matches the linear curve at inputs 0.0, 0.5, and 1.0
    The derivative of the easing function is zero at the boundaries.
    """
    v_out = (1 - cos(pi_std*v_in))/2


# some of the more exotic trig functions

def sinc[dtype: DType](f: Rad[Scalar[dtype]], out result: Scalar[dtype])
    where dtype.is_floating_point():

    if f == 0.0:
        result = 1
    elif f >= 0.01:
        result = f.sin()/f.value
    else:
        # TODO: what's this? some kind of approximation? taylor series maybe?
        var g = f.value*f.value
        result = 1 - g/6 + g*g/120
