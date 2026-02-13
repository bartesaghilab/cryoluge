
from cryoluge.image import Vec
from cryoluge.math.units import Rad, Deg, pi


struct EulerAnglesZYZ[dtype: DType](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    """Three Euler angles, in the ZYZ convention, in radians."""
    var psi: Rad[dtype]
    var theta: Rad[dtype]
    var phi: Rad[dtype]

    @staticmethod
    fn zero(out self: Self):
        self = Self(fill=Rad[dtype](0))

    fn __init__(out self, *, fill: Rad[dtype]):
        self.psi = fill
        self.theta = fill
        self.phi = fill

    fn __init__(out self, *, psi: Rad[dtype], theta: Rad[dtype], phi: Rad[dtype]):
        self.psi = psi
        self.theta = theta
        self.phi = phi

    fn __init__(out self, *, psi: Deg[dtype], theta: Deg[dtype], phi: Deg[dtype]):
        self.psi = psi.to_rad()
        self.theta = theta.to_rad()
        self.phi = phi.to_rad()

    fn __init__(out self, *, from_mat: Matrix.D3[dtype], match_csp1: Bool = False):
        ref m = from_mat

        if match_csp1:
            # TEMP: implement the old csp1 behavior, for reference
            if m[2,2] < 1:
                if m[2,2] > -1:
                    self.theta = Rad[dtype].acos(m[2,2])
                    var sin_theta = self.theta.sin()
                    self.psi = Rad[dtype].atan2(m[2,1]/sin_theta, m[2,0]/sin_theta)
                    self.phi = Rad[dtype].atan2(m[1,2]/sin_theta, -m[0,2]/sin_theta)
                else:
                    self.theta = pi[dtype]
                    self.psi = Rad[dtype](0)
                    self.phi = Rad[dtype].atan2(-m[0,1], -m[0,0])
            else:
                self.theta = Rad[dtype](0)
                self.psi = Rad[dtype](0)
                self.phi = Rad[dtype].atan2(m[0,1], m[0,0])

        else:
            # otherwise, do the usual method for Euler angle decomposition
            self.theta = Rad[dtype].acos(m[2,2])
            if abs(m[0,2]) < 1e-4:
                self.psi = Rad[dtype].atan2(m[1,0], m[1,1])
                self.phi = Rad[dtype](0)
            else:
                self.psi = Rad[dtype].atan2(m[2,1], -m[2,0])
                self.phi = Rad[dtype].atan2(m[1,2], m[0,2])

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "EulerAnglesZYZ[psi=", self.psi.to_deg(),
            "°, theta=", self.theta.to_deg(),
            "°, phi=", self.phi.to_deg(), "°]"
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn to_matrix(self, mut mat: Matrix.D3[dtype]):
        var rot_psi = Matrix.D3[dtype](rotate_z=self.psi)
        var rot_theta = Matrix.D3[dtype](rotate_y=self.theta)
        var rot_phi = Matrix.D3[dtype](rotate_z=self.phi)
        mat = rot_phi*rot_theta*rot_psi

    fn normalize(self, out normalized: Self):
        normalized = Self(
            psi = self.psi.normalize(),
            theta = self.theta.normalize(),
            phi = self.phi.normalize()
        )

    fn normalize_positive(self, out normalized: Self):
        normalized = Self(
            psi = self.psi.normalize_positive(),
            theta = self.theta.normalize_positive(),
            phi = self.phi.normalize_positive()
        )

    # math functions

    fn __neg__(self, out result: Self):
        result = Self(
            psi = -self.psi,
            theta = -self.theta,
            phi = -self.phi
        )

    fn __add__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi + other.psi,
            theta = self.theta + other.theta,
            phi = self.phi + other.phi
        )

    fn __iadd__(mut self, other: Self):
        self.psi += other.psi
        self.theta += other.theta
        self.phi += other.phi

    fn __sub__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi - other.psi,
            theta = self.theta - other.theta,
            phi = self.phi - other.phi
        )

    fn __isub__(mut self, other: Self):
        self.psi -= other.psi
        self.theta -= other.theta
        self.phi -= other.phi

    fn __mul__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi * other.psi,
            theta = self.theta * other.theta,
            phi = self.phi * other.phi
        )

    fn __imul__(mut self, other: Self):
        self.psi *= other.psi
        self.theta *= other.theta
        self.phi *= other.phi

    fn __mul__(self, other: Scalar[dtype], out result: Self):
        result = self * Self(fill=other)

    fn __rmul__(self, other: Scalar[dtype], out result: Self):
        result = self * other

    fn __imul__(mut self, other: Scalar[dtype]):
        self *= Self(fill=other)

    fn __truediv__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi / other.psi,
            theta = self.theta / other.theta,
            phi = self.phi / other.phi
        )

    fn __itruediv__(mut self, other: Self):
        self.psi /= other.psi
        self.theta /= other.theta
        self.phi /= other.phi

    fn __truediv__(self, other: Scalar[dtype], out result: Self):
        result = self / Self(fill=other)

    fn __rtruediv__(self, other: Scalar[dtype], out result: Self):
        result = self / other

    fn __itruediv__(mut self, other: Scalar[dtype]):
        self /= Self(fill=other)

    fn abs(self, out result: Self):
        result = Self(
            psi = self.psi.abs(),
            theta = self.theta.abs(),
            phi = self.phi.abs()
        )

    fn sum(self, out result: Rad[dtype]):
        result = self.psi + self.theta + self.phi

    fn dists(self, other: Self, out dists: Self):
        dists = Self(
            psi = self.psi.dist(other.psi),
            theta = self.theta.dist(other.theta),
            phi = self.phi.dist(other.phi)
        )

    # conversions

    fn map[
        out_dtype: DType,
        //,
        mapper: fn(Rad[dtype]) capturing -> Rad[out_dtype]
    ](self, out result: EulerAnglesZYZ[out_dtype]):
        result = EulerAnglesZYZ[out_dtype](
            psi=mapper(self.psi),
            theta=mapper(self.theta),
            phi=mapper(self.phi)
        )

    fn map_scalar[out_dtype: DType](self, out result: EulerAnglesZYZ[out_dtype]):
        @parameter
        fn func(v: Rad[dtype], out mapped: Rad[out_dtype]):
            mapped = Rad[out_dtype](v)
        result = self.map[mapper=func]()

    fn map_float32(self, out result: EulerAnglesZYZ[DType.float32]):
        result = self.map_scalar[DType.float32]()

    fn map_float64(self, out result: EulerAnglesZYZ[DType.float64]):
        result = self.map_scalar[DType.float64]()
