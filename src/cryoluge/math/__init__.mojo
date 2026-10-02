
from std.math import sin, cos, pi, min, max
from std.complex import ComplexScalar, ComplexSIMD

from .dimension import *
from .vec import *
from .matrix import *
from .euler import *
from .rand import *
from .stats import *
from .sphere import *
from .box import *


def is_even(n: Int) -> Bool:
    # TODO: does this work for negative values?
    # TODO: TEST!!
    return (n & 1) == 0


def is_odd(n: Int) -> Bool:
    return not is_even(n)


def round[digits: Int, dtype: DType, w: Int](v: SIMD[dtype,w]) -> SIMD[dtype,w]:
    # the SIMD.__round__(digits) implementation is slow
    # this is a lot faster, when we know the number of digits at compile-time
    comptime if digits == 0:
        return v
    else:
        comptime f = SIMD[dtype,w]( 10**digits )
        comptime oof = 1.0/f
        return (v*f).__round__()*oof

    # TODO: would a bitwise right shift then left shift accomplish something similar?


def clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    min: SIMD[dtype,width]
):
    r = std.math.max(n, min)


def clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    max: SIMD[dtype,width]
):
    r = std.math.min(n, max)


def clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    min: SIMD[dtype,width],
    max: SIMD[dtype,width]
):
    r = std.math.min(std.math.max(n, min), max)


def ladder[width: Int](out v: SIMD[DType.int,width]):
    """
    Returns a vector containing all the integers in the range [0,width) in order.
    """
    v = SIMD[DType.int,width]()
    comptime for i in range(width):
        v[i] = i


def ease_linear[dtype: DType, width: Int](v_in: SIMD[dtype,width], out v_out: SIMD[dtype,width]):
    """
    Easing function over [0,1] with a linear shape, essentially a no-op.
    """
    v_out = v_in


def ease_cos[dtype: DType, width: Int](v_in: SIMD[dtype,width], out v_out: SIMD[dtype,width])
    where dtype.is_floating_point():
    """
    Easing function over [0,1] with a cos shape.
    This curve matches the linear curve at inputs 0.0, 0.5, and 1.0
    The derivative of the easing function is zero at the boundaries.
    """
    v_out = (1 - cos(pi*v_in))/2


# some of the more exotic trig functions

def sinc[dtype: DType](f: Scalar[dtype], out result: Scalar[dtype])
    where dtype.is_floating_point():

    if f == 0:
        result = 1
    elif f >= 0.01:
        result = sin(f)/f
    else:
        # TODO: what's this? some kind of approximation? taylor series maybe?
        var g = f*f
        result = 1 - g/6 + g*g/120
