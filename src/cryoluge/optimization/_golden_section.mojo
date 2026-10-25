
from std.math import sqrt

from cryoluge.optimization import ObjectiveInfo, ObjectiveLine, Coord, Value, LineSearch, LineSearchResult


def golden_section[info: ObjectiveInfo](
    *,
    objective: Some[ObjectiveLine[info]],
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

    # first iteration:

    # sample points in the interval
    comptime phi = Coord[info]( (1.0 + sqrt(5.0))/2 )  # ~1.618
    comptime invphi = 1.0/phi  # ~0.618
    var step = (i_max - i_min)*invphi
    var x_down = i_max - step
    var x_up = i_min + step
    var f_down = objective(x_down)
    var f_up = objective(x_up)

    # refine the interval based on the samples
    var next_is_down = f_down < f_up
    if next_is_down:
        i_max = x_up
        x_up = x_down
        f_up = f_down
    else:
        i_min = x_down
        x_down = x_up
        f_down = f_up

    # do the rest of the iterations until convergence
    while i_max - i_min > min_interval_width:
        
        # sample the next point
        step = (i_max - i_min)*invphi
        if next_is_down:
            x_down = i_max - step
            f_down = objective(x_down)
        else:
            x_up = i_min + step
            f_up = objective(x_up)

        # update the interval
        next_is_down = f_down < f_up
        if next_is_down:
            i_max = x_up
            x_up = x_down
            f_up = f_down
        else:
            i_min = x_down
            x_down = x_up
            f_down = f_up
    
    # the interval is small now, just pick the center
    var x = (i_min + i_max)/2
    result = LineSearchResult[info](
        x = x,
        fx = objective(x)
    )


@fieldwise_init
struct GoldenSectionLineSearch[
    dtype: DType
](
    Copyable,
    Movable,
    LineSearch
):
    var min_interval_width: Scalar[Self.dtype]

    def minimize[info: ObjectiveInfo](
        self,
        *,
        objective: Some[ObjectiveLine[info]],
        x_start: Coord[info],  # NOTE: not used
        x_min: Coord[info],
        x_max: Coord[info],
        fx_start: Optional[Value[info]] = None,  # NOTE: not used
        out result: LineSearchResult[info]
    ):
        result = golden_section(
            objective = objective,
            x_min = x_min,
            x_max = x_max,
            min_interval_width = Coord[info](self.min_interval_width)
        )
