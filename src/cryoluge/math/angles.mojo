
from math import pi


fn deg_to_rad[dtype: DType](*, deg: Scalar[dtype], out rad: Scalar[dtype]):
    rad = deg/180*pi


fn rad_to_deg[dtype: DType](*, rad: Scalar[dtype], out deg: Scalar[dtype]):
    deg = rad*180/pi


# TODO: angle normalization functions?
