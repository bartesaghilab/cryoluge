
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.cistem import Block, BlockedReader, Parameter, ParameterType
from cryoluge.collections import KeyableSet


struct Blocks:
    alias particles = Block(15, "particles")
    alias tilts = Block(35, "tilts")

    alias _all = List[Block](
        Self.particles,
        Self.tilts
    )

    @staticmethod
    fn all() -> List[Block]:
        return materialize[Self._all]()


struct ExtraParameters:
    alias p_ind = Parameter(15, "p_ind", ParameterType.int)
    alias t_ind = Parameter(35, "t_ind", ParameterType.int)
    alias r_ind = Parameter(70, "r_ind", ParameterType.int)

    alias p_shift_x = Parameter(3, "p_shift_x", ParameterType.float)
    alias p_shift_y = Parameter(9, "p_shift_y", ParameterType.float)
    alias p_shift_z = Parameter(27, "p_shift_z", ParameterType.float)
    alias p_psi = Parameter(81, "p_psi", ParameterType.float)
    alias p_theta = Parameter(273, "p_theta", ParameterType.float)
    alias p_phi = Parameter(819, "p_phi", ParameterType.float)
    alias original_x_position_3d = Parameter(2457, "original_x_position_3d", ParameterType.float)
    alias original_y_position_3d = Parameter(7371, "original_y_position_3d", ParameterType.float)
    alias original_z_position_3d = Parameter(22113, "original_z_position_3d", ParameterType.float)
    alias p_score = Parameter(66339, "p_score", ParameterType.float)
    alias p_occ = Parameter(199017, "p_occ", ParameterType.float)

    alias t_shift_x = Parameter(7, "t_shift_x", ParameterType.float)
    alias t_shift_y = Parameter(49, "t_shift_y", ParameterType.float)
    alias t_tilt_ang = Parameter(343, "t_tilt_ang", ParameterType.float)
    alias t_tilt_axis = Parameter(2401, "t_tilt_axis", ParameterType.float)

    alias _all = List[Parameter](
        Self.p_ind,
        Self.t_ind,
        Self.r_ind,
        Self.p_shift_x,
        Self.p_shift_y,
        Self.p_shift_z,
        Self.p_psi,
        Self.p_theta,
        Self.p_phi,
        Self.original_x_position_3d,
        Self.original_y_position_3d,
        Self.original_z_position_3d,
        Self.p_score,
        Self.p_occ,
        Self.t_shift_x,
        Self.t_shift_y,
        Self.t_tilt_ang,
        Self.t_tilt_axis
    )

    @staticmethod
    fn all() -> List[Parameter]:
        return materialize[Self._all]()


alias TEST_FILE = "test/cryoluge/cistem/test.blocked.cistem"
# NOTE: parameters in this file:
# particles(15)
#    0: p_pind(15,int)
#    1: p_shift_x(3,float)
#    2: p_shift_y(9,float)
#    3: p_shift_z(27,float)
#    4: p_psi(81,float)
#    5: p_theta(273,float)
#    6: p_phi(819,float)
#    7: original_x_position_3d(2457,float)
#    8: original_y_position_3d(7371,float)
#    9: original_z_position_3d(22113,float)
#   10: p_score(66339,float)
#   11: p_occ(199017,float)
# tilts(35)
#    0: t_ind(35,int)
#    1: r_ind(70,int)
#    2: t_shift_x(7,float)
#    3: t_shift_y(49,float)
#    4: t_tilt_ang(343,float)
#    5: t_tilt_axis(2401,float)


def test_blocks():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var blocked_reader = BlockedReader(file_reader)

        assert_equal(len(blocked_reader.blocks()), 2)

        assert_equal(blocked_reader.blocks()[0], Blocks.particles)
        assert_equal(blocked_reader.blocks()[1], Blocks.tilts)


def test_cols_and_lines():
    with open(TEST_FILE, "r") as f:
        var file_reader = FileReader(f)
        var blocked_reader = BlockedReader(
            file_reader,
            blocks=Blocks.all(),
            parameters=ExtraParameters.all()
        )

        # read the particles block
        ref particles_reader = blocked_reader.seek(Blocks.particles)

        assert_equal(len(particles_reader.cols()), 12)
        assert_equal(particles_reader.num_lines(), 2)

        # just spot-check a few parameters
        particles_reader.read_line()
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_ind](), 0)
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_shift_x](), -14.362735)
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_occ](), 100.0)
        particles_reader.read_line()
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_ind](), 1)
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_shift_x](), 19.996538)
        assert_equal(particles_reader.get_parameter[ExtraParameters.p_occ](), 100.0)

        # read the tilts block
        ref tilts_reader = blocked_reader.seek(Blocks.tilts)

        assert_equal(len(tilts_reader.cols()), 6)
        assert_equal(tilts_reader.num_lines(), 2)

        # just spot-check a few parameters
        tilts_reader.read_line()
        assert_equal(tilts_reader.get_parameter[ExtraParameters.t_ind](), 1)
        assert_equal(tilts_reader.get_parameter[ExtraParameters.r_ind](), 2)
        assert_equal(tilts_reader.get_parameter[ExtraParameters.t_tilt_axis](), 0.0)
        tilts_reader.read_line()
        assert_equal(tilts_reader.get_parameter[ExtraParameters.t_ind](), 3)
        assert_equal(tilts_reader.get_parameter[ExtraParameters.r_ind](), 4)
        assert_equal(tilts_reader.get_parameter[ExtraParameters.t_tilt_axis](), 0.0)
