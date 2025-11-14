
from math import pi


fn deg_to_rad[dtype: DType, size: Int](*, deg: SIMD[dtype,size], out rad: SIMD[dtype,size]):
    rad = deg/180*pi


fn rad_to_deg[dtype: DType, size: Int](*, rad: SIMD[dtype,size], out deg: SIMD[dtype,size]):
    deg = rad*180/pi


# TODO: angle normalization functions?
