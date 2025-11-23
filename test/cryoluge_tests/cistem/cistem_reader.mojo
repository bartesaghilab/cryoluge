
from testing import assert_equal

from cryoluge.lang import rebind_scalar
from cryoluge.io import FileReader
from cryoluge.cistem import Reader, CistemParameters, Parameter, ParameterType


comptime funcs = __functions_in_module()


struct ExtraParameters:
    comptime im_ind = Parameter(20, "im_ind", ParameterType.int)
    comptime p_ind = Parameter(15, "p_ind", ParameterType.int)
    comptime t_ind = Parameter(35, "t_ind", ParameterType.int)
    comptime r_ind = Parameter(70, "r_ind", ParameterType.int)
    comptime f_ind = Parameter(55, "f_ind", ParameterType.int)
    comptime fshift_x = Parameter(11, "fshift_x", ParameterType.float)
    comptime fshift_y = Parameter(121, "fshift_y", ParameterType.float)

    comptime _all = List[Parameter](
        Self.im_ind,
        Self.p_ind,
        Self.t_ind,
        Self.r_ind,
        Self.f_ind,
        Self.fshift_x,
        Self.fshift_y
    )

    @staticmethod
    fn all() -> List[Parameter]:
        return materialize[Self._all]()


comptime TEST_FILE = "test/cryoluge_tests/cistem/test.cistem"
# NOTE: parameters in this file:
#    0: position_in_stack(1,uint)
#    1: psi(4,float),
#    2: theta(4194304,float)
#    3: phi(8388608,float)
#    4: x_shift(8,float)
#    5: y_shift(16,float)
#    6: defocus_1(32,float)
#    7: defocus_2(64,float)
#    8: defocus_angle(128,float)
#    9: phase_shift(256,float)
#   10: image_is_active(2,int)
#   11: occupancy(512,float)
#   12: logp(1024,float)
#   13: sigma(2048,float)
#   14: score(4096,float)
#   15: pixel_size(16384,float)
#   16: microscope_voltage(32768,float)
#   17: microscope_cs(65536,float)
#   18: amplitude_contrast(131072,float)
#   19: beam_tilt_x(262144,float)
#   20: beam_tilt_y(524288,float)
#   21: image_shift_x(1048576,float)
#   22: image_shift_y(2097152,float)
#   23: original_x_position(8589934592,float)
#   24: original_y_position(17179869184,float)
#   25: im_ind(20,int)
#   26: p_ind(15,int)
#   27: t_ind(35,int)
#   28: r_ind(70,int)
#   29: f_ind(55,int)
#   30: fshift_x(11,float)
#   31: fshift_y(121,float)


def test_default_parameters():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)
        assert_equal(len(cistem_reader.cols()), 32)

        # spot-check a few parameters
        assert_equal(cistem_reader.cols()[0], CistemParameters.position_in_stack)
        assert_equal(cistem_reader.cols()[11], CistemParameters.occupancy)
        assert_equal(cistem_reader.cols()[24], CistemParameters.original_y_position)

        # should have a few unknown parameters
        assert_equal(cistem_reader.cols()[25], Parameter.unknown(20, ParameterType.int))
        assert_equal(cistem_reader.cols()[31], Parameter.unknown(121, ParameterType.float))


def test_some_default_parameters():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader, parameters=[
            CistemParameters.psi,
            CistemParameters.theta,
            CistemParameters.phi
        ])
        assert_equal(len(cistem_reader.cols()), 32)

        # the explicit parameters should show up
        assert_equal(cistem_reader.cols()[1], CistemParameters.psi)
        assert_equal(cistem_reader.cols()[2], CistemParameters.theta)
        assert_equal(cistem_reader.cols()[3], CistemParameters.phi)

        # the others should be unknown
        assert_equal(cistem_reader.cols()[0], Parameter.unknown(1, ParameterType.uint))
        assert_equal(cistem_reader.cols()[31], Parameter.unknown(121, ParameterType.float))


def test_extended_parameters():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader, parameters=ExtraParameters.all())
        assert_equal(len(cistem_reader.cols()), 32)

        # spot-check a few parameters
        assert_equal(cistem_reader.cols()[25], ExtraParameters.im_ind)
        assert_equal(cistem_reader.cols()[30], ExtraParameters.fshift_x)
        assert_equal(cistem_reader.cols()[31], ExtraParameters.fshift_y)


def test_has_parameter():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader, parameters=[
            CistemParameters.psi,
            CistemParameters.theta,
            CistemParameters.phi
        ])
        assert_equal(len(cistem_reader.cols()), 32)

        # the ones we declared should show up
        assert_equal(cistem_reader.has_parameter[CistemParameters.psi](), True)
        assert_equal(cistem_reader.has_parameter[CistemParameters.theta](), True)
        assert_equal(cistem_reader.has_parameter[CistemParameters.phi](), True)

        # cistem parameters should show up, even if we didn't declare them
        assert_equal(cistem_reader.has_parameter[CistemParameters.position_in_stack](), True)

        # others won't be there
        comptime rando = Parameter(Int64(5), "rando", ParameterType.double)
        assert_equal(cistem_reader.has_parameter[rando](), False)


def test_get_parameter_first_line():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)
        cistem_reader.read_line()

        # spot-check a few values
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 1)
        assert_equal(cistem_reader.get_parameter[CistemParameters.psi](), 89.76255)
        assert_equal(cistem_reader.get_parameter[CistemParameters.theta](), 117.066574)
        assert_equal(cistem_reader.get_parameter[CistemParameters.phi](), 213.39417)
        assert_equal(cistem_reader.get_parameter[CistemParameters.original_x_position](), 3817.0)
        assert_equal(cistem_reader.get_parameter[CistemParameters.original_y_position](), 260.0)


def test_get_parameter_into_string():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)
        cistem_reader.read_line()

        # spot-check a few values
        assert_equal(cistem_reader.get_parameter_into_string(CistemParameters.position_in_stack), '1')
        assert_equal(cistem_reader.get_parameter_into_string(CistemParameters.psi), '89.76255')


def test_read_a_few_lines():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)

        assert_equal(cistem_reader.next_line(), 0)
        assert_equal(cistem_reader.eof(), False)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 1)
        assert_equal(cistem_reader.next_line(), 1)
        assert_equal(cistem_reader.eof(), False)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 2)
        assert_equal(cistem_reader.next_line(), 2)
        assert_equal(cistem_reader.eof(), False)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 3)
        assert_equal(cistem_reader.next_line(), 3)
        assert_equal(cistem_reader.eof(), False)


def test_seek():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)

        cistem_reader.seek(2)
        assert_equal(cistem_reader.next_line(), 2)
        assert_equal(cistem_reader.eof(), False)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 3)

        cistem_reader.seek(cistem_reader.num_lines() - 1)
        assert_equal(cistem_reader.next_line(), cistem_reader.num_lines() - 1)
        assert_equal(cistem_reader.eof(), False)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 29)
        assert_equal(cistem_reader.next_line(), cistem_reader.num_lines())
        assert_equal(cistem_reader.eof(), True)


def test_get_parameter_types():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var cistem_reader = Reader(file_reader)

        cistem_reader.read_line()

        # you can use the full parameter type expression if you want,
        # but it's not type-compatible with plain ol' UInt32 =(
        var p1: Scalar[CistemParameters.position_in_stack.type.dtype.value()] = cistem_reader.get_parameter[CistemParameters.position_in_stack]()
        assert_equal(p1, 1)

        # but you can explicitly rebind it
        var u1: UInt32 = rebind_scalar[DType.uint32](p1)
        assert_equal(u1, 1)

        # or you can supply the concrete dtype
        # and the API will do the rebind for you automatically
        var p2: UInt32 = cistem_reader.get_parameter[CistemParameters.position_in_stack, DType.uint32]()
        assert_equal(p2, 1)

        # or you can use one of the typed functions for easy-mode
        var p3: UInt32 = cistem_reader.get_parameter_uint[CistemParameters.position_in_stack]()
        assert_equal(p3, 1)

        # check some of the other typed functions too
        # (at leas the ones present in the test file)
        var p4: Int32 = cistem_reader.get_parameter_int[CistemParameters.image_is_active]()
        assert_equal(p4, 0)
        var p5: Float32 = cistem_reader.get_parameter_float[CistemParameters.pixel_size]()
        assert_equal(p5, 8.4)
