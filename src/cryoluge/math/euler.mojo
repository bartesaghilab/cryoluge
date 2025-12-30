
from math import sin, acos, asin, atan2, pi

from cryoluge.image import Vec
from cryoluge.math import normalize_minus_pi_to_pi, normalize_0_to_2pi


struct EulerAnglesZYZ[dtype: DType](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    """Three Euler angles, in the ZYZ convention, in radians."""
    var psi_rad: Scalar[dtype]
    var theta_rad: Scalar[dtype]
    var phi_rad: Scalar[dtype]

    @staticmethod
    fn zero(out self: Self):
        self = Self(psi_rad=0, theta_rad=0, phi_rad=0)

    fn __init__(out self, *, fill: Scalar[dtype]):
        self.psi_rad = fill
        self.theta_rad = fill
        self.phi_rad = fill

    fn __init__(out self, *, psi_rad: Scalar[dtype], theta_rad: Scalar[dtype], phi_rad: Scalar[dtype]):
        self.psi_rad = psi_rad
        self.theta_rad = theta_rad
        self.phi_rad = phi_rad

    fn __init__(out self, *, psi_deg: Scalar[dtype], theta_deg: Scalar[dtype], phi_deg: Scalar[dtype]):
        self.psi_rad = deg_to_rad(deg=psi_deg)
        self.theta_rad = deg_to_rad(deg=theta_deg)
        self.phi_rad = deg_to_rad(deg=phi_deg)

    fn __init__(out self, *, from_mat: Matrix.D3[dtype]):
        ref m = from_mat
        self.theta_rad = acos(m[2,2])
        if abs(m[0,2]) < 1e-4:
            self.psi_rad = atan2(m[1,0], m[1,1])
            self.phi_rad = 0
        else:
            self.psi_rad = atan2(m[2,1], -m[2,0])
            self.phi_rad = atan2(m[1,2], m[0,2])

    # TEMP: implement the old csp1 behavior, for reference
    fn __init__(out self, *, from_mat_csp1: Matrix.D3[dtype]):
        ref m = from_mat_csp1
        if m[2,2] < 1:
            if m[2,2] > -1:
                self.theta_rad = acos(m[2,2])
                var sin_theta_rad = sin(self.theta_rad)
                self.psi_rad = atan2(
                    m[2,1]/sin_theta_rad,
                    m[2,0]/sin_theta_rad
                )
                self.phi_rad = atan2(
                    m[1,2]/sin_theta_rad,
                    -m[0,2]/sin_theta_rad
                )
            else:
                self.theta_rad = pi
                self.psi_rad = 0
                self.phi_rad = atan2(-m[0,1], -m[0,0])
        else:
            self.theta_rad = 0
            self.psi_rad = 0
            self.phi_rad = atan2(m[0,1], m[0,0])

    fn psi_deg(self, out deg: Scalar[dtype]):
        deg = rad_to_deg(rad=self.psi_rad)

    fn theta_deg(self, out deg: Scalar[dtype]):
        deg = rad_to_deg(rad=self.theta_rad)

    fn phi_deg(self, out deg: Scalar[dtype]):
        deg = rad_to_deg(rad=self.phi_rad)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "EulerAnglesZYZ[psi=", self.psi_deg(),
            "°, theta=", self.theta_deg(),
            "°, phi=", self.phi_deg(), "°]"
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn to_matrix(self, mut mat: Matrix.D3[dtype]):
        var rot_psi = Matrix.D3[dtype](rotate_z_rad=self.psi_rad)
        var rot_theta = Matrix.D3[dtype](rotate_y_rad=self.theta_rad)
        var rot_phi = Matrix.D3[dtype](rotate_z_rad=self.phi_rad)
        mat = rot_phi*rot_theta*rot_psi

    fn normalize(mut self):
        self.psi_rad = normalize_minus_pi_to_pi(rad=self.psi_rad)
        self.theta_rad = normalize_minus_pi_to_pi(rad=self.theta_rad)
        self.phi_rad = normalize_minus_pi_to_pi(rad=self.phi_rad)

    fn normalize_positive(mut self):
        self.psi_rad = normalize_0_to_2pi(rad=self.psi_rad)
        self.theta_rad = normalize_0_to_2pi(rad=self.theta_rad)
        self.phi_rad = normalize_0_to_2pi(rad=self.phi_rad)

    # math functions

    fn __neg__(self, out result: Self):
        result = Self(
            psi_rad = -self.psi_rad,
            theta_rad = -self.theta_rad,
            phi_rad = -self.phi_rad
        )

    fn __add__(self, other: Self, out result: Self):
        result = Self(
            psi_rad = self.psi_rad + other.psi_rad,
            theta_rad = self.theta_rad + other.theta_rad,
            phi_rad = self.phi_rad + other.phi_rad
        )

    fn __iadd__(mut self, other: Self):
        self.psi_rad += other.psi_rad
        self.theta_rad += other.theta_rad
        self.phi_rad += other.phi_rad

    fn __sub__(self, other: Self, out result: Self):
        result = Self(
            psi_rad = self.psi_rad - other.psi_rad,
            theta_rad = self.theta_rad - other.theta_rad,
            phi_rad = self.phi_rad - other.phi_rad
        )

    fn __isub__(mut self, other: Self):
        self.psi_rad -= other.psi_rad
        self.theta_rad -= other.theta_rad
        self.phi_rad -= other.phi_rad

    fn __mul__(self, other: Self, out result: Self):
        result = Self(
            psi_rad = self.psi_rad * other.psi_rad,
            theta_rad = self.theta_rad * other.theta_rad,
            phi_rad = self.phi_rad * other.phi_rad
        )

    fn __imul__(mut self, other: Self):
        self.psi_rad *= other.psi_rad
        self.theta_rad *= other.theta_rad
        self.phi_rad *= other.phi_rad

    fn __mul__(self, other: Scalar[dtype], out result: Self):
        result = self * Self(fill=other)

    fn __rmul__(self, other: Scalar[dtype], out result: Self):
        result = self * other

    fn __imul__(mut self, other: Scalar[dtype]):
        self *= Self(fill=other)

    fn __truediv__(self, other: Self, out result: Self):
        result = Self(
            psi_rad = self.psi_rad / other.psi_rad,
            theta_rad = self.theta_rad / other.theta_rad,
            phi_rad = self.phi_rad / other.phi_rad
        )

    fn __itruediv__(mut self, other: Self):
        self.psi_rad /= other.psi_rad
        self.theta_rad /= other.theta_rad
        self.phi_rad /= other.phi_rad

    fn __truediv__(self, other: Scalar[dtype], out result: Self):
        result = self / Self(fill=other)

    fn __rtruediv__(self, other: Scalar[dtype], out result: Self):
        result = self / other

    fn __itruediv__(mut self, other: Scalar[dtype]):
        self /= Self(fill=other)

    fn abs(self, out result: Self):
        result = Self(
            psi_rad = abs(self.psi_rad),
            theta_rad = abs(self.theta_rad),
            phi_rad = abs(self.phi_rad)
        )

    fn sum_rad(self, out result: Scalar[dtype]):
        result = self.psi_rad + self.theta_rad + self.phi_rad

    fn sum_deg(self, out result: Scalar[dtype]):
        result = self.psi_deg() + self.theta_deg() + self.phi_deg()

    fn dists(self, other: Self, out dists: Self):
        dists = Self(
            psi_rad = angle_dist(rad_a=self.psi_rad, rad_b=other.psi_rad),
            theta_rad = angle_dist(rad_a=self.theta_rad, rad_b=other.theta_rad),
            phi_rad = angle_dist(rad_a=self.phi_rad, rad_b=other.phi_rad)
        )

    # conversions

    fn map_rad[
        out_dtype: DType,
        //,
        mapper: fn(Scalar[dtype]) capturing -> Scalar[out_dtype]
    ](self, out result: EulerAnglesZYZ[out_dtype]):
        result = EulerAnglesZYZ[out_dtype](
            psi_rad=mapper(self.psi_rad),
            theta_rad=mapper(self.theta_rad),
            phi_rad=mapper(self.phi_rad)
        )

    fn map_rad_scalar[out_dtype: DType](self, out result: EulerAnglesZYZ[out_dtype]):
        @parameter
        fn func(v: Scalar[dtype], out mapped: Scalar[out_dtype]):
            mapped = Scalar[out_dtype](v)
        result = self.map_rad[mapper=func]()

    fn map_float32(self, out result: EulerAnglesZYZ[DType.float32]):
        result = self.map_rad_scalar[DType.float32]()

    fn map_float64(self, out result: EulerAnglesZYZ[DType.float64]):
        result = self.map_rad_scalar[DType.float64]()
