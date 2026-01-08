

fn ccd[
    info: ObjectiveInfo,
    LS: LineSearch,
    //,
    objective: Objective[info]
](
    *,
    line_search: LS,
    x_start: Coords[info],
    x_min: Coords[info],
    x_max: Coords[info],
    max_iterations: Int,
    value_threshold: Value[info],
    out result: OptimizationResult[info]
):
    var x_here = x_start.copy()
    var x_next = x_here.copy()

    # start with the current value
    var fx_here = objective(x_here)
    var fx_next = Value[info](0)

    for _ in range(max_iterations):

        @parameter
        for d in range(info.dim.rank):

            @parameter
            fn line(x: Coord[info], out fx: Value[info]):
                x_next[d] = x
                fx = objective(x_next)

            var line_result = line_search.minimize[line=line](
                x_start = x_here[d],
                x_min = x_min[d],
                x_max = x_max[d]
            )
            x_next[d] = line_result.x
            fx_next = line_result.fx

        # did the value get any better?
        var improvement = fx_here - fx_next
        if improvement > 0:
            
            # yup: keep it
            x_here = x_next.copy()
            fx_here = fx_next

            # but stop early if we level off
            if improvement < value_threshold:
                break

        else:
            # nope: we're done here
            break

    result = OptimizationResult(
        x = x_here^,
        fx = fx_here
    )


@fieldwise_init
struct CCDMinimizer[
    dtype: DType,
    LS: LineSearch & Copyable & Movable
](
    Copyable,
    Movable,
    Minimizer
):
    var max_iterations: Int
    var value_threshold: Scalar[dtype]
    var line_search: LS

    fn minimize[
        info: ObjectiveInfo,
        //,
        objective: Objective[info]
    ](
        self,
        x_start: Coords[info],
        x_min: Coords[info],
        x_max: Coords[info],
        out result: OptimizationResult[info]
    ):
        result = ccd[objective](
            line_search = self.line_search,
            x_start = x_start,
            x_min = x_min,
            x_max = x_max,
            max_iterations = self.max_iterations,
            value_threshold = Value[info](self.value_threshold)
        )
