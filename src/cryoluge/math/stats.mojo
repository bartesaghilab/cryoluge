
from cryoluge.math import Dimension, Vec
from cryoluge.math.units import Unit, UnitType


struct MeanVariance[dim: Dimension, dtype: DType, utype: UnitType=UnitType._None](
    Copyable,
    Movable
):
    var mean: Vec[Unit[utype,dtype],dim]
    var variance: Vec[Unit[utype,dtype],dim]

    comptime D1 = MeanVariance[Dimension.D1,_,_]
    comptime D2 = MeanVariance[Dimension.D2,_,_]
    comptime D3 = MeanVariance[Dimension.D3,_,_]

    fn __init__(out self):
        self.mean = Vec[Unit[utype,dtype],dim](fill=Unit[utype,dtype](0))
        self.variance = Vec[Unit[utype,dtype],dim](fill=Unit[utype,dtype](0))

    fn add(mut self, v: Vec[Unit[utype,dtype],dim]):
        self.mean += v
        self.variance += v**2
    
    fn normalize(mut self, count: Int):
        self.mean /= count
        self.variance /= count
        self.variance -= self.mean**2
