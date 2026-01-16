

fn powell[
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
    # https://en.wikipedia.org/wiki/Powell%27s_method
    # https://phys.uri.edu/nigh/NumRec/bookfpdf/f10-5.pdf
    # technically, this implementation is a modified version of Powell's method,
    # since Powell's original method tends to introduce linear dependence in the
    # directions, which are supposed to stay orthogonal

    # initialize the directions to be the axis-aligned orthonormal basis
    var directions = List[Coords[info]](capacity=info.dim.rank)
    @parameter
    for d in range(info.dim.rank):
        var dir = Coords[info](fill=0)
        dir[d] = Coord[info](1)
        directions.append(dir^)    

    # evaluate the starting point
    var x_here = x_start.copy()
    var fx_here = objective(x_here)

    var x_prev = x_here.copy()

    # print("Powell!")  # DEBUG

    for _ in range(max_iterations):

        var fx_before_dirs = fx_here
        var best_d = 0
        var best_improvement = Value[info](0)

        # print("\titer: x=", x_here, "fx=", fx_here)  # DEBUG

        # search along each direction
        for d in range(info.dim.rank):
            var fx_before_dir = fx_here

            @parameter
            fn line_d(x: Coord[info], out fx: Value[info]):
                fx = objective(x_here + directions[d]*x)

            # do line search along direction d
            var xd_min, xd_max = _line_bounds(
                x = x_here,
                dir = directions[d],
                x_min = x_min,
                x_max = x_max
            )
            var line_result = line_search.minimize[line=line_d](
                x_start = Coord[info](0),
                x_min = xd_min,
                x_max = xd_max,
                fx_start = fx_here
            )
            x_here += directions[d]*line_result.x
            fx_here = line_result.fx

            # track the direction with the best improvement
            var improvement = fx_before_dir - fx_here
            if improvement > best_improvement:
                best_improvement = improvement
                best_d = d

            # print("\t\td=", d, "t=", line_result.x, "in [", xd_min, ",", xd_max, "] x=", x_here, "fx=", fx_here, "imp=", improvement, "best_d=", best_d)  # DEBUG

        # stop early if we didn't improve
        if fx_before_dirs - fx_here < value_threshold:
            # print("\tstall out!")  # DEBUG
            break

        # calculate the overall direction of this iteration
        var iter_delta = x_here - x_prev
        var iter_dir = iter_delta/iter_delta.len()
        x_prev = x_here.copy()

        # print("\textra step: delta=", iter_delta, "dir=", iter_dir)  # DEBUG

        # take another step in the iteration direction (with bounds checking)
        var xd_min, xd_max = _line_bounds(
            x = x_here,
            dir = iter_dir,
            x_min = x_min,
            x_max = x_max
        )
        var x_extrapolated = x_here + iter_delta*min(max(xd_min, xd_max), Coord[info](1))
        var fx_extrapolated = objective(x_extrapolated)

        # print("\textrapolated: t=", min(max(xd_min, xd_max), Coord[info](1)), "in [", xd_min, ",", xd_max, "] x=", x_extrapolated, "fx=", fx_extrapolated)  # DEBUG

        # if that's not worse than before the iteration ...
        if fx_extrapolated < fx_before_dirs:

            # and if the new direction has any potential for further minimization ...
            # i.e.: 2(f0 - 2fN + fe)( (f0 - fN) - df )^2 < df(f0 - fe)^2
            var left = (fx_before_dirs - fx_here*2 + fx_extrapolated)
                *(fx_before_dirs - fx_here - best_improvement)**2
                *2
            var right = best_improvement*(fx_before_dirs - fx_extrapolated)**2
            if left < right:

                @parameter
                fn line_iter(x: Coord[info], out fx: Value[info]):
                    fx = objective(x_here + iter_dir*x)

                # do line search along the iteration direction
                var line_result = line_search.minimize[line=line_iter](
                    x_start = Coord[info](0),
                    x_min = xd_min,
                    x_max = xd_max,
                    fx_start = fx_here
                )
                x_here += iter_dir*line_result.x
                fx_here = line_result.fx

                # print("\textra line search x=", x_here, "fx=", fx_here)  # DEBUG

                # if we got any improvement in this direction, replace the previous best direction
                if line_result.x != 0:
                    directions[best_d] = iter_dir^

    result = OptimizationResult(
        x = x_here^,
        fx = fx_here
    )


fn _line_bounds[info: ObjectiveInfo](
    x: Coords[info],
    dir: Coords[info],  # should be normalized
    x_min: Coords[info],
    x_max: Coords[info],
    out bounds: Tuple[Coord[info],Coord[info]]
):
    var best_t_pos = Coord[info].MAX
    var best_t_neg = Coord[info].MIN

    @parameter
    for d in range(info.dim.rank):

        # skip dimensions with no movement
        if dir[d] == 0:
            continue

        # do the plane intersections to get the distances in 1D space
        var t_min = (x_min[d] - x[d])/dir[d]
        var t_max = (x_max[d] - x[d])/dir[d]

        # sort the bounds
        if t_min > t_max:
            var swap = t_min
            t_min = t_max
            t_max = swap

        # track the closest intersections
        if t_min > best_t_neg:
            best_t_neg = t_min
        if t_max < best_t_pos:
            best_t_pos = t_max

    bounds = (best_t_neg, best_t_pos)


@fieldwise_init
struct PowellMinimizer[
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
        result = powell[objective](
            line_search = self.line_search,
            x_start = x_start,
            x_min = x_min,
            x_max = x_max,
            max_iterations = self.max_iterations,
            value_threshold = Value[info](self.value_threshold)
        )
