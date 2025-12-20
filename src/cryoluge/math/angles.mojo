
from math import pi


fn deg_to_rad[dtype: DType, size: Int](*, deg: SIMD[dtype,size], out rad: SIMD[dtype,size]):
    rad = deg/180*pi


fn rad_to_deg[dtype: DType, size: Int](*, rad: SIMD[dtype,size], out deg: SIMD[dtype,size]):
    deg = rad*180/pi


fn normalize_minus_pi_to_pi[dtype: DType, size: Int](*, rad: SIMD[dtype,size], out result: SIMD[dtype,size]):
    result = rad
    while result < -pi:
        result += 2*pi
    while result > pi:
        result -= 2*pi


fn angle_dist[dtype: DType, size: Int](*, rad_a: SIMD[dtype,size], rad_b: SIMD[dtype,size], out dist: SIMD[dtype,size]):
    var norm_a = normalize_minus_pi_to_pi(rad=rad_a)
    var norm_b = normalize_minus_pi_to_pi(rad=rad_b)
    dist = abs(norm_a - norm_b)
    if dist > pi:
        dist = pi - dist


# TODO: use units?
# TODO: overloads of trig fns for Deg,Rad units?
