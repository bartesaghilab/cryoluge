
from cryoluge.math.units import Unit, UnitType, pi


struct Sphere[utype: UnitType, dtype: DType]:

    @staticmethod
    def radius(*, volume: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return (0.75/pi[dtype].value*volume)**(1.0/3.0)

    @staticmethod
    def radius(*, diameter: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return diameter/2
    
    @staticmethod
    def diameter(*, radius: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return radius*2

    @staticmethod
    def cross_area(*, radius: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return pi[dtype].value*radius**2

    @staticmethod
    def cross_area(*, diameter: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return Self.cross_area(radius=Self.radius(diameter=diameter))

    @staticmethod
    def surface_area(*, radius: Unit[utype,dtype]) -> Unit[utype,dtype]:
        return 4*pi[dtype].value*radius**2
