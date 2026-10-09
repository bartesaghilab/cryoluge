
from cryoluge.math.units import Unit, UnitType, pi


struct Sphere[dtype: DType]
    where dtype.is_floating_point():

    comptime V = Scalar[Self.dtype]

    @staticmethod
    def radius(*, volume: Self.V) -> Self.V:
        return (0.75/pi[Self.dtype].value*volume)**(1.0/3.0)

    @staticmethod
    def radius(*, diameter: Self.V) -> Self.V:
        return diameter/2
    
    @staticmethod
    def diameter(*, radius: Self.V) -> Self.V:
        return radius*2

    @staticmethod
    def cross_area(*, radius: Self.V) -> Self.V:
        return pi[Self.dtype].value*radius**2

    @staticmethod
    def cross_area(*, diameter: Self.V) -> Self.V:
        return Self.cross_area(radius=Self.radius(diameter=diameter))

    @staticmethod
    def surface_area(*, radius: Self.V) -> Self.V:
        return 4*pi[Self.dtype].value*radius**2
