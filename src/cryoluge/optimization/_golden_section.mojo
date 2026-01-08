
from math import sqrt


fn golden_section[
    info: ObjectiveInfo,
    //,
    objective_line: ObjectiveLine[info]
](
    *,
    x_min: Coord[info],
    x_max: Coord[info],
    min_interval_width: Coord[info],
    out result: LineSearchResult[info]
):
    # https://en.wikipedia.org/wiki/Golden_section_search
    # simple, easy, works ok ... enough

    # initialize the interval
    var i_min = x_min
    var i_max = x_max

    while i_max - i_min > min_interval_width:

        # sample points from the interval using the golden ratio
        comptime invphi = Scalar[info.dtype_coord](2/(sqrt(5.0) + 1))  # ~0.618
        var step = (i_max - i_min)*invphi
        var x_down = i_max - step
        var x_up = i_min + step

        # refine the interval based on the samples
        var f_down = objective_line(x_down)
        var f_up = objective_line(x_up)
        if f_down < f_up:
            i_max = x_up
        else:
            i_min = x_down

    # the interval is small now, just pick the center
    var x = (i_min + i_max)/2
    result = LineSearchResult[info](
        x = x,
        fx = objective_line(x)
    )


@fieldwise_init
struct GoldenSectionLineSearch[
    dtype: DType
](
    Copyable,
    Movable,
    LineSearch
):
    var min_interval_width: Scalar[dtype]

    fn minimize[
        info: ObjectiveInfo,
        //,
        line: ObjectiveLine[info]
    ](
        self,
        x_start: Coord[info],  # NOTE: not used
        x_min: Coord[info],
        x_max: Coord[info],
        out result: LineSearchResult[info]
    ):
        result = golden_section[line](
            x_min = x_min,
            x_max = x_max,
            min_interval_width = Coord[info](self.min_interval_width)
        )
