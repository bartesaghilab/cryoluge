
from cryoluge.cistem import Parameter, ParameterType, Block


struct ExtraParameters_DELETEME:
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


struct ExtraParameters:
    comptime im_ind = Parameter(20, "im_ind", ParameterType.int)
    comptime p_ind = Parameter(15, "p_ind", ParameterType.int)
    comptime t_ind = Parameter(35, "t_ind", ParameterType.int)
    comptime r_ind = Parameter(70, "r_ind", ParameterType.int)
    comptime f_ind = Parameter(55, "f_ind", ParameterType.int)

    comptime p_shift_x = Parameter(3, "p_shift_x", ParameterType.float)
    comptime p_shift_y = Parameter(9, "p_shift_y", ParameterType.float)
    comptime p_shift_z = Parameter(27, "p_shift_z", ParameterType.float)
    comptime p_psi = Parameter(81, "p_psi", ParameterType.float)
    comptime p_theta = Parameter(273, "p_theta", ParameterType.float)
    comptime p_phi = Parameter(819, "p_phi", ParameterType.float)
    comptime original_x_position_3d = Parameter(2457, "original_x_position_3d", ParameterType.float)
    comptime original_y_position_3d = Parameter(7371, "original_y_position_3d", ParameterType.float)
    comptime original_z_position_3d = Parameter(22113, "original_z_position_3d", ParameterType.float)
    comptime p_score = Parameter(66339, "p_score", ParameterType.float)
    comptime p_occ = Parameter(199017, "p_occ", ParameterType.float)

    comptime t_shift_x = Parameter(7, "t_shift_x", ParameterType.float)
    comptime t_shift_y = Parameter(49, "t_shift_y", ParameterType.float)
    comptime t_tilt_ang = Parameter(343, "t_tilt_ang", ParameterType.float)
    comptime t_tilt_axis = Parameter(2401, "t_tilt_axis", ParameterType.float)

    comptime fshift_x = Parameter(11, "fshift_x", ParameterType.float)
    comptime fshift_y = Parameter(121, "fshift_y", ParameterType.float)

    comptime _all = List[Parameter](
        Self.im_ind,
        Self.p_ind,
        Self.t_ind,
        Self.r_ind,
        Self.f_ind,
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
        Self.t_tilt_axis,
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


struct Blocks:
    comptime particles = Block(15, "particles")
    comptime tilts = Block(35, "tilts")

    comptime _all = List[Block](
        Self.particles,
        Self.tilts
    )

    @staticmethod
    fn all() -> List[Block]:
        return materialize[Self._all]()


comptime TEST_FILE_EXT = "test/cryoluge_tests/cistem/test.blocked.cistem"
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
