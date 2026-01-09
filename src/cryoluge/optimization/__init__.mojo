
from cryoluge.math import Dimension, Vec

from ._golden_section import *
from ._brent import *
from ._ccd import *


@fieldwise_init
struct ObjectiveInfo:
    var dtype_coord: DType
    var dtype_value: DType
    var dim: Dimension

    fn d1(self, out info_1d: ObjectiveInfo):
        info_1d = Self(self.dtype_coord, self.dtype_value, Dimension.D1)


comptime Coord[info: ObjectiveInfo] = Scalar[info.dtype_coord]
comptime Coords[info: ObjectiveInfo] = Vec[Coord[info],info.dim]
comptime Value[info: ObjectiveInfo] = Scalar[info.dtype_value]


comptime Objective[info: ObjectiveInfo] = fn(
    x: Coords[info],
    out fx: Value[info]
) capturing


@fieldwise_init
struct OptimizationResult[info: ObjectiveInfo](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var x: Coords[info]
    var fx: Value[info]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('OptimizationResult[x=', self.x, ', fx=', self.fx, ']')

    fn __str__(self) -> String:
        return String.write(self)


trait Minimizer:
    fn minimize[
        info: ObjectiveInfo,
        //,
        objective: Objective[info]
    ](
        self,
        *,
        x_start: Coords[info],
        x_min: Coords[info],
        x_max: Coords[info],
        out result: OptimizationResult[info]
    ): ...


comptime ObjectiveLine[info: ObjectiveInfo] = fn(
    x: Coord[info],
    out fx: Value[info]
) capturing


@fieldwise_init
struct LineSearchResult[info: ObjectiveInfo](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var x: Coord[info]
    var fx: Value[info]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('LineSearchResult[x=', self.x, ', fx=', self.fx, ']')

    fn __str__(self) -> String:
        return String.write(self)


trait LineSearch:
    fn minimize[
        info: ObjectiveInfo,
        //,
        line: ObjectiveLine[info]
    ](
        self,
        *,
        x_start: Coord[info],
        x_min: Coord[info],
        x_max: Coord[info],
        out result: LineSearchResult[info]
    ): ...
