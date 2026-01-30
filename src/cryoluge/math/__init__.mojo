
from math import sin, cos, pi
from complex import ComplexSIMD

from .dimension import *
from .vec import *
from .angles import *
from .matrix import *
from .euler import *


# NOTE: coming in mojo stdlib in next release
comptime ComplexScalar = ComplexSIMD[_,1]


fn is_even(n: Int) -> Bool:
    # TODO: does this work for negative values?
    # TODO: TEST!!
    return (n & 1) == 0


fn is_odd(n: Int) -> Bool:
    return not is_even(n)


fn clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    min: SIMD[dtype,width]
):
    r = math.max(n, min)


fn clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    max: SIMD[dtype,width]
):
    r = math.min(n, max)


fn clamp[dtype: DType, width: Int](
    n: SIMD[dtype,width],
    out r: SIMD[dtype,width],
    *,
    min: SIMD[dtype,width],
    max: SIMD[dtype,width]
):
    r = math.min(math.max(n, min), max)


fn ease_linear[dtype: DType, width: Int](v_in: SIMD[dtype,width], out v_out: SIMD[dtype,width]):
    """
    Easing function over [0,1] with a linear shape, essentially a no-op.
    """
    v_out = v_in


fn ease_cos[dtype: DType, width: Int](v_in: SIMD[dtype,width], out v_out: SIMD[dtype,width]):
    """
    Easing function over [0,1] with a cos shape.
    This curve matches the linear curve at inputs 0.0, 0.5, and 1.0
    The derivative of the easing function is zero at the boundaries.
    """
    v_out = (1 - cos(pi*v_in))/2


# some of the more exotic trig functions

fn sinc[dtype: DType](f: Scalar[dtype], out result: Scalar[dtype]):
    if f == 0:
        result = 1
    elif f >= 0.01:
        result = sin(f)/f
    else:
        # TODO: what's this? some kind of approximation? taylor series maybe?
        var g = f*f
        result = 1 - g/6 + g*g/120
