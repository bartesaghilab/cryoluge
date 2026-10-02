
from cryoluge.math import Vec
from cryoluge.math.units import Unit, UnitType


struct MeanVariance[dim: Int, dtype: DType, utype: UnitType=UnitType._None](
    Copyable,
    Movable
):
    var mean: Vec[dim,Unit[utype,dtype]]
    var variance: Vec[dim,Unit[utype,dtype]]

    def __init__(out self):
        self.mean = Vec[dim,Unit[utype,dtype]](fill=Unit[utype,dtype](0))
        self.variance = Vec[dim,Unit[utype,dtype]](fill=Unit[utype,dtype](0))

    def add(mut self, v: Vec[dim,Unit[utype,dtype]]):
        self.mean += v
        self.variance += v**2
    
    def normalize(mut self, count: Int):
        self.mean /= count
        self.variance /= count
        self.variance -= self.mean**2
