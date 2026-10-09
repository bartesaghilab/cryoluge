
from cryoluge.math import Vec, Matrix
from cryoluge.math.units import Rad, Deg, pi


struct EulerAnglesZYZ[dtype: DType](
    Copyable,
    Movable,
    Deinitable,
    Writable
) where dtype.is_floating_point():
    """Three Euler angles, in the ZYZ convention, in radians."""

    var psi: Rad[Self.V]
    var theta: Rad[Self.V]
    var phi: Rad[Self.V]

    comptime V = Scalar[Self.dtype]

    @staticmethod
    def zero(out self: Self):
        self = Self(fill=Rad[Self.V](0))

    def __init__(out self, *, fill: Rad[Self.V]):
        self.psi = fill
        self.theta = fill
        self.phi = fill

    def __init__(out self, *, psi: Rad[Self.V], theta: Rad[Self.V], phi: Rad[Self.V]):
        self.psi = psi
        self.theta = theta
        self.phi = phi

    def __init__(out self, *, psi: Deg[Self.V], theta: Deg[Self.V], phi: Deg[Self.V]):
        self.psi = psi.to_rad()
        self.theta = theta.to_rad()
        self.phi = phi.to_rad()

    def __init__(out self, *, from_mat: Matrix[3,3,Self.dtype], match_csp1: Bool = False):
        ref m = from_mat

        if match_csp1:
            # TEMP: implement the old csp1 behavior, for reference
            if m[2,2] < 1:
                if m[2,2] > -1:
                    self.theta = Rad[Self.V].acos(m[2,2])
                    var sin_theta = self.theta.sin()
                    self.psi = Rad[Self.V].atan2(m[2,1]/sin_theta, m[2,0]/sin_theta)
                    self.phi = Rad[Self.V].atan2(m[1,2]/sin_theta, -m[0,2]/sin_theta)
                else:
                    self.theta = pi[Self.dtype]
                    self.psi = Rad[Self.V](0)
                    self.phi = Rad[Self.V].atan2(-m[0,1], -m[0,0])
            else:
                self.theta = Rad[Self.V](0)
                self.psi = Rad[Self.V](0)
                self.phi = Rad[Self.V].atan2(m[0,1], m[0,0])

        else:
            # otherwise, do the usual method for Euler angle decomposition
            self.theta = Rad[Self.V].acos(m[2,2])
            if abs(m[0,2]) < 1e-4:
                self.psi = Rad[Self.V].atan2(m[1,0], m[1,1])
                self.phi = Rad[Self.V](0)
            else:
                self.psi = Rad[Self.V].atan2(m[2,1], -m[2,0])
                self.phi = Rad[Self.V].atan2(m[1,2], m[0,2])

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "EulerAnglesZYZ[psi=", self.psi.to_deg(),
            "°, theta=", self.theta.to_deg(),
            "°, phi=", self.phi.to_deg(), "°]"
        )

    def to_matrix(self, mut mat: Matrix[3,3,Self.dtype]):
        var rot_psi = Matrix[3,3,Self.dtype](rotate_z=self.psi)
        var rot_theta = Matrix[3,3,Self.dtype](rotate_y=self.theta)
        var rot_phi = Matrix[3,3,Self.dtype](rotate_z=self.phi)
        mat = rot_phi*rot_theta*rot_psi

    def normalize(self, out normalized: Self):
        normalized = Self(
            psi = self.psi.normalize(),
            theta = self.theta.normalize(),
            phi = self.phi.normalize()
        )

    def normalize_positive(self, out normalized: Self):
        normalized = Self(
            psi = self.psi.normalize_positive(),
            theta = self.theta.normalize_positive(),
            phi = self.phi.normalize_positive()
        )

    # math functions

    def __neg__(self, out result: Self):
        result = Self(
            psi = -self.psi,
            theta = -self.theta,
            phi = -self.phi
        )

    def __add__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi + other.psi,
            theta = self.theta + other.theta,
            phi = self.phi + other.phi
        )

    def __iadd__(mut self, other: Self):
        self.psi += other.psi
        self.theta += other.theta
        self.phi += other.phi

    def __sub__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi - other.psi,
            theta = self.theta - other.theta,
            phi = self.phi - other.phi
        )

    def __isub__(mut self, other: Self):
        self.psi -= other.psi
        self.theta -= other.theta
        self.phi -= other.phi

    def __mul__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi * other.psi,
            theta = self.theta * other.theta,
            phi = self.phi * other.phi
        )

    def __imul__(mut self, other: Self):
        self.psi *= other.psi
        self.theta *= other.theta
        self.phi *= other.phi

    def __mul__(self, other: Self.V, out result: Self):
        result = self * Self(fill=Rad(other))

    def __rmul__(self, other: Self.V, out result: Self):
        result = self * other

    def __imul__(mut self, other: Self.V):
        self *= Self(fill=Rad(other))

    def __truediv__(self, other: Self, out result: Self):
        result = Self(
            psi = self.psi / other.psi,
            theta = self.theta / other.theta,
            phi = self.phi / other.phi
        )

    def __itruediv__(mut self, other: Self):
        self.psi /= other.psi
        self.theta /= other.theta
        self.phi /= other.phi

    def __truediv__(self, other: Self.V, out result: Self):
        result = self / Self(fill=Rad(other))

    def __rtruediv__(self, other: Self.V, out result: Self):
        result = self / other

    def __itruediv__(mut self, other: Self.V):
        self /= Self(fill=Rad(other))

    def abs(self, out result: Self):
        result = Self(
            psi = self.psi.abs(),
            theta = self.theta.abs(),
            phi = self.phi.abs()
        )

    def sum(self, out result: Rad[Self.V]):
        result = self.psi + self.theta + self.phi

    def dists(self, other: Self, out dists: Self):
        dists = Self(
            psi = self.psi.dist(other.psi),
            theta = self.theta.dist(other.theta),
            phi = self.phi.dist(other.phi)
        )

    # conversions

    def to_dtype[out_dtype: DType](self, out result: EulerAnglesZYZ[out_dtype])
        where out_dtype.is_floating_point():

        result = EulerAnglesZYZ(
            psi = self.psi.to_dtype[out_dtype](),
            theta = self.theta.to_dtype[out_dtype](),
            phi = self.phi.to_dtype[out_dtype]()
        )
