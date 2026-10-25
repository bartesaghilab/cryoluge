
from cryoluge.math import Sphere
from cryoluge.math.units import KDa, Ang


struct SphericalParticle[dtype: DType](
    Copyable,
    Movable
) where dtype.is_floating_point():

    var mass_kda: KDa[Scalar[Self.dtype]]
    var volume_a: Ang[Scalar[Self.dtype]]
    var radius_a: Ang[Scalar[Self.dtype]]

    comptime Sphere = Sphere[Self.dtype]

    def __init__(out self, *, mass_kda: KDa[Scalar[Self.dtype]]):
        self.mass_kda = mass_kda

        # estimate the volume of a globular protein with the given mass
        # TODO: where does this model come from?
        self.volume_a = Ang(mass_kda.value*1000/0.81)

        # use the usual sphere formulas to find out the rest of the geometry
        self.radius_a = Self.Sphere.radius(volume=self.volume_a.value)

    def cross_area_a(self) -> Ang[Scalar[Self.dtype]]:
        return Self.Sphere.cross_area(radius=self.radius_a.value)
    
    def diameter_a(self) -> Ang[Scalar[Self.dtype]]:
        return Self.Sphere.diameter(radius=self.radius_a.value)
