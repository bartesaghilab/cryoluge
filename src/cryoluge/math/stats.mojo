
from cryoluge.math.vec import Vec
from cryoluge.math.units import Unit, UnitType


struct MeanVariance[dim: Int, dtype: DType](
    Copyable,
    Movable
) where dim > 0:

    var mean: Self.V
    var variance: Self.V

    comptime V = Vec[Self.dim,Self.dtype]

    def __init__(out self):
        self.mean = Self.V(fill=0)
        self.variance = Self.V(fill=0)

    def add(mut self, v: Self.V):
        self.mean += v
        self.variance += v**2
    
    def normalize(mut self, count: Int):
        var count_dt = Scalar[Self.dtype](count)
        self.mean /= count_dt
        self.variance /= count_dt
        self.variance -= self.mean**2
