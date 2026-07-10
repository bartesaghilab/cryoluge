
from cryoluge.math import Sphere
from cryoluge.math.units import KDa, Ang


struct SphericalParticle[dtype: DType](
    Copyable,
    Movable
):
    var mass_kda: KDa[dtype]
    var volume_a: Ang[dtype]
    var radius_a: Ang[dtype]

    comptime Sphere = Sphere[Ang.utype,dtype]

    fn __init__(out self, *, mass_kda: KDa[dtype]):
        self.mass_kda = mass_kda

        # estimate the volume of a globular protein with the given mass
        # TODO: where does this model come from?
        self.volume_a = Ang(mass_kda.value*1000/0.81)

        # use the usual sphere formulas to find out the rest of the geometry
        self.radius_a = Self.Sphere.radius(volume=self.volume_a)

    fn cross_area_a(self) -> Ang[dtype]:
        return Self.Sphere.cross_area(radius=self.radius_a)
    
    fn diameter_a(self) -> Ang[dtype]:
        return Self.Sphere.diameter(radius=self.radius_a)
