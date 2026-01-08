
from testing import assert_equal, assert_true, assert_false

from cryoluge.math import Dimension, Vec
from cryoluge.math.error import err_abs
from cryoluge.optimization import ObjectiveInfo, Coord, Value, golden_section, GoldenSectionLineSearch

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
    fn f(x: Coord[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x - 5)**2 + 42 )

    var result = golden_section[f](
        x_min = -10,
        x_max = 10,
        min_interval_width = 1e-2
    )
    # NOTE: due to f32 precision and taking the square, the best we can do is 2 digits of precision

    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x), 5, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)


def test_line_search():

    comptime info = ObjectiveInfo(
        dtype_coord = dtype,
        dtype_value = dtype,
        dim = Dimension.D1
    )
    @parameter
    fn f(x: Coord[info], out fx: Value[info]):
        fx = rebind[Value[info]]( (x - 5)**2 + 42 )

    var result = GoldenSectionLineSearch[dtype](
        min_interval_width = 1e-2
    ).minimize[line=f](
        x_start = 0,
        x_min = -10,
        x_max = 10
    )

    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.x), 5, eps=1e-2)
    assert_equal_float[err_fn](rebind[Scalar[dtype]](result.fx), 42, eps=1e-2)
