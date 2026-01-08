
from testing import assert_equal, assert_true, assert_false

from cryoluge.math import Dimension, Vec
from cryoluge.math.error import err_abs
from cryoluge.optimization import ObjectiveInfo, Coord, Coords, Value, GoldenSectionLineSearch, ccd, CCDMinimizer

from cryoluge_testlib import assert_equal_float


comptime funcs = __functions_in_module()


comptime dtype = DType.float32
comptime err_fn = err_abs[dtype]


def test_quadratic():

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = Dimension.D1
    )
    @parameter
    fn f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + 42 )

    var result = ccd[f](
        line_search = GoldenSectionLineSearch(
            min_interval_width = 1e-2
        ),
        x_start = Coords[info](x=0),
        x_min = Coords[info](x=-10),
        x_max = Coords[info](x=10),
        max_iterations = 10,
        value_threshold = 1e-2
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x), 5, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)


def test_biquadratic():

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = Dimension.D2
    )
    @parameter
    fn f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + (x[1] + 7)**2 + 42 )

    var result = ccd[f](
        line_search = GoldenSectionLineSearch(
            min_interval_width = 1e-2
        ),
        x_start = Coords[info](fill=0),
        x_min = Coords[info](fill=-10),
        x_max = Coords[info](fill=10),
        max_iterations = 10,
        value_threshold = 1e-2
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x[0]), 5, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x[1]), -7, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)


def test_minimizer():

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = Dimension.D2
    )
    @parameter
    fn f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + (x[1] + 7)**2 + 42 )

    var result = CCDMinimizer[dtype](
        line_search = GoldenSectionLineSearch(
            min_interval_width = 1e-2
        ),
        max_iterations = 10,
        value_threshold = 1e-2
    ).minimize[objective=f](
        x_start = Coords[info](fill=0),
        x_min = Coords[info](fill=-10),
        x_max = Coords[info](fill=10)
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x[0]), 5, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x[1]), -7, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)
