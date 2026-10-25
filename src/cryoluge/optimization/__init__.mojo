
from cryoluge.math import Vec

from ._golden_section import *
from ._brent import *
from ._ccd import *
from ._powell import *


@fieldwise_init
struct ObjectiveInfo:
    var dtype_coord: DType
    var dtype_value: DType
    var dim: Int

    def d1(self, out info_1d: ObjectiveInfo):
        info_1d = Self(self.dtype_coord, self.dtype_value, 1)


comptime Coord[info: ObjectiveInfo] = Scalar[info.dtype_coord]
comptime Coords[info: ObjectiveInfo] = Vec[info.dim,info.dtype_coord]
comptime Value[info: ObjectiveInfo] = Scalar[info.dtype_value]


comptime Objective[info: ObjectiveInfo] = def(
    x: Coords[info],
    out fx: Value[info]
)


@fieldwise_init
struct OptimizationResult[info: ObjectiveInfo](
    Copyable,
    Movable,
    Writable
):
    var x: Coords[Self.info]
    var fx: Value[Self.info]

    def write_to[W: Writer](self, mut writer: W):
        writer.write('OptimizationResult[x=', self.x, ', fx=', self.fx, ']')


trait Minimizer:
    def minimize[info: ObjectiveInfo](
        self,
        *,
        objective: Some[Objective[info]],
        x_start: Coords[info],
        x_min: Coords[info],
        x_max: Coords[info],
        out result: OptimizationResult[info]
    ): ...


comptime ObjectiveLine[info: ObjectiveInfo] = def(
    x: Coord[info],
    out fx: Value[info]
)


@fieldwise_init
struct LineSearchResult[info: ObjectiveInfo](
    Copyable,
    Movable,
    Writable
):
    var x: Coord[Self.info]
    var fx: Value[Self.info]

    def write_to[W: Writer](self, mut writer: W):
        writer.write('LineSearchResult[x=', self.x, ', fx=', self.fx, ']')


trait LineSearch:
    def minimize[info: ObjectiveInfo](
        self,
        *,
        objective: Some[ObjectiveLine[info]],
        x_start: Coord[info],
        x_min: Coord[info],
        x_max: Coord[info],
        fx_start: Optional[Value[info]] = None,
        out result: LineSearchResult[info]
    ): ...
