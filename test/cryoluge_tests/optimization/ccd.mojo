
from testing import assert_equal, assert_true, assert_false

from cryoluge.math import Vec
from cryoluge.math.error import err_abs
from cryoluge.optimization import ObjectiveInfo, Coord, Coords, Value, BrentLineSearch, ccd, CCDMinimizer

from cryoluge.test import assert_equal_float

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


comptime dtype = DType.float32
comptime err_fn = err_abs[dtype]


def test_quadratic() raises:

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = 1
    )
    def f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + 42 )

    var result = ccd[info](
        objective = f,
        line_search = BrentLineSearch(
            max_iterations = 10,
            min_interval_width = 1e-2
        ),
        x_start = Coords[info](x=0),
        x_min = Coords[info](x=-10),
        x_max = Coords[info](x=10),
        max_iterations = 10,
        value_threshold = 1e-2
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float(err_fn, rebind[Vec[1,dtype]](result.x), [5], eps=1e-2)
    assert_equal_float(err_fn, rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)


def test_biquadratic() raises:

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = 2
    )
    def f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + (x[1] + 7)**2 + 42 )

    var result = ccd[info](
        objective = f,
        line_search = BrentLineSearch(
            max_iterations = 10,
            min_interval_width = 1e-2
        ),
        x_start = Coords[info](fill=0),
        x_min = Coords[info](fill=-10),
        x_max = Coords[info](fill=10),
        max_iterations = 10,
        value_threshold = 1e-2
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float(err_fn, rebind[Vec[2,dtype]](result.x), [5, -7], eps=1e-2)
    assert_equal_float(err_fn, rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)


def test_minimizer() raises:

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = 2
    )
    def f(x: Coords[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x[0] - 5)**2 + (x[1] + 7)**2 + 42 )

    var result = CCDMinimizer[dtype](
        line_search = BrentLineSearch(
            max_iterations = 10,
            min_interval_width = 1e-2
        ),
        max_iterations = 10,
        value_threshold = 1e-2
    ).minimize[info](
        objective = f,
        x_start = Coords[info](fill=0),
        x_min = Coords[info](fill=-10),
        x_max = Coords[info](fill=10)
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float(err_fn, rebind[Vec[2,dtype]](result.x), [5, -7], eps=1e-2)
    assert_equal_float(err_fn, rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)
