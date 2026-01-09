

fn brent[
    info: ObjectiveInfo,
    //,
    objective_line: ObjectiveLine[info]
](
    *,
    x_start: Coord[info],
    x_min: Coord[info],
    x_max: Coord[info],
    max_iterations: Int,
    min_interval_width: Coord[info],
    out result: LineSearchResult[info]
):

    # https://people.math.wisc.edu/~chr/am205/g_act/am205_workshop_optimization.pdf
    # https://cs.fit.edu/~dmitra/SciComp/20Fall/Optimization_Brent's_Method.pdf

    # initialize the interval
    var i_min = x_min  # a
    var i_max = x_max  # b
    
    # initialize the samples
    var x_lo = x_start  # w
    var x_mid = x_start  # x
    var x_hi = x_start  # v
    var f_lo = objective_line(x_lo)
    var f_mid = objective_line(x_mid)
    var f_hi = objective_line(x_hi)

    # print('brent:')  # TEMP

    for _ in range(max_iterations):

        # TEMP
        # print('\tIteration:', i)
        # print('\t\tI=[', i_min, ',', i_max, ']')
        # print('\t\tx=[', x_lo, ', ', x_mid, ', ', x_hi, ']')
        # print('\t\tf=[', f_lo, ', ', f_mid, ', ', f_hi, ']')

        # check for convergence
        if (i_max - i_min) < min_interval_width:
            break

        # pick the next step: try to do parabolic interpolation
        var step: Coord[info]

        var (num,denom) = _parabolic_step(
            x0=x_lo,
            x1=x_mid,
            x2=x_hi,
            fx0=f_lo,
            fx1=f_mid,
            fx2=f_hi
        )
        if denom != 0:
            
            # yup: should work
            step = num/denom/2

            # print('\t\tparabolic step=', step, 'num=', num, 'denom=', denom)  # TEMP

            # make sure the step is in range
            x_next = x_lo + step
            # print('\t\t\tcheck x=', x_next)  # TEMP
            if x_next < i_min or x_next > i_max:

                # nope: get the next point using golden section instead
                step = _golden_section_step[info](x_mid=x_lo, i_min=i_min, i_max=i_max)

                # print('\t\t\tout of range! golden section step=', step)  # TEMP

        else:

            # nope: get the next point using golden section instead
            step = _golden_section_step[info](x_mid=x_lo, i_min=i_min, i_max=i_max)

            # print('\t\tgolden section step=', step)  # TEMP

        var x_next = x_lo + step
        var f_next = objective_line(x_next)

        # print('\t\tnext', 'x=', x_next, 'f=', f_next)  # TEMP

        # update the interval
        if f_next > f_lo:
            # got worse
            if step < 0:
                i_min = x_next
            else:
                i_max = x_next
            # update sample points: keep f_lo < f_mid < f_hi
            if (f_next <= f_mid) or (x_mid == x_lo):
                x_hi = x_mid
                f_hi = f_mid
                x_mid = x_next
                f_mid = f_next
            elif (f_next <= f_hi) or (x_hi == x_lo) or (x_hi == x_mid):
                x_hi = x_next
                f_hi = f_next
        else:
            # got better
            if step < 0:
                i_max = x_lo
            else:
                i_min = x_lo
            # update sample points: keep f_lo < f_mid < f_hi
            x_hi = x_mid
            f_hi = f_mid
            x_mid = x_lo
            f_mid = f_lo
            x_lo = x_next
            f_lo = f_next

    result = LineSearchResult[info](
        x = x_lo,
        fx = f_lo
    )


fn _parabolic_step[info: ObjectiveInfo](
    *,
    x0: Coord[info],
    x1: Coord[info],
    x2: Coord[info],
    fx0: Value[info],
    fx1: Value[info],
    fx2: Value[info],
    out ratio: Tuple[Coord[info],Coord[info]]
):
    var x0mx1 = x0 - x1
    var x0mx2 = x0 - x2
    var fx0mfx2 = Coord[info](fx0 - fx2)
    var fx0mfx1 = Coord[info](fx0 - fx1)

    var num = (x0mx2**2)*fx0mfx1 - (x0mx1**2)*fx0mfx2
    var denom = x0mx1*fx0mfx2 - x0mx2*fx0mfx1

    ratio = (num, denom)


fn _golden_section_step[info: ObjectiveInfo](
    *,
    x_mid: Coord[info],
    i_min: Coord[info],
    i_max: Coord[info],
    out step: Coord[info]
):
    comptime phi = Coord[info]( (1.0 + sqrt(5.0))/2 )  # ~1.618
    comptime two_m_phi = 2.0 - phi  # ~0.382

    var i_mid = (i_min + i_max)/2
    if x_mid < i_mid:
        step = two_m_phi*(i_max - x_mid)
    else:
        step = two_m_phi*(i_min - x_mid)  # NOTE: this is negative


@fieldwise_init
struct BrentLineSearch[
    dtype: DType
](
    Copyable,
    Movable,
    LineSearch
):
    var max_iterations: Int
    var min_interval_width: Scalar[dtype]

    fn minimize[
        info: ObjectiveInfo,
        //,
        line: ObjectiveLine[info]
    ](
        self,
        x_start: Coord[info],
        x_min: Coord[info],
        x_max: Coord[info],
        out result: LineSearchResult[info]
    ):
        result = brent[line](
            x_start = x_start,
            x_min = x_min,
            x_max = x_max,
            max_iterations = self.max_iterations,
            min_interval_width = Coord[info](self.min_interval_width)
        )
