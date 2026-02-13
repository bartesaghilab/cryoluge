
from testing import assert_equal

from cryoluge.io import FileReader
from cryoluge.cistem import BlockedReader


comptime funcs = __functions_in_module()


def test_blocks():
    with open(TEST_FILE_EXT, "r") as f:
        var file_reader = FileReader(f)
        var blocked_reader = BlockedReader(file_reader)

        assert_equal(len(blocked_reader.blocks()), 2)

        assert_equal(blocked_reader.blocks()[0], Blocks.particles)
        assert_equal(blocked_reader.blocks()[1], Blocks.tilts)


def test_cols_and_lines():
    with open(TEST_FILE_EXT, "r") as f:
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
