
from testing import assert_equal

from cryoluge.lang import LexicalScope
from cryoluge.io import ByteBuffer, BytesWriter, BytesReader, FileReader
from cryoluge.cistem import BlockedReader, BlockedWriter, CistemParameters


comptime funcs = __functions_in_module()


def test_write_blocks():

    comptime block_first = Block(5, "First")
    comptime cols_first = [
        CistemParameters.position_in_stack,  # uint => UInt32 => 4
        CistemParameters.psi,  # float => Float32 => 4
        ExtraParameters.p_ind,  # int => Int32 => 4
    ]

    comptime block_second = Block(7, "Second")
    comptime cols_second = [
        ExtraParameters.t_ind,
        ExtraParameters.p_psi
    ]

    var buf = ByteBuffer(125)
    # block first: 8 + 4 + 4 + 3*(8+1) + 2*(4+4+4) = 67
    # block second: 8 + 4 + 4 + 2*(8+1) + 3*(4+4) = 58

    with LexicalScope():
        var writer = BytesWriter(buf.span())
        var blocked_writer = BlockedWriter(writer)

        with LexicalScope():
            var cistem_writer = blocked_writer.write_block(
                block_first,
                num_lines = 2,
                cols = materialize[cols_first]()
            )

            cistem_writer.set_parameter[CistemParameters.position_in_stack](5)
            cistem_writer.set_parameter[CistemParameters.psi](4.2)
            cistem_writer.set_parameter[ExtraParameters.p_ind](9)
            cistem_writer.write_line()

            cistem_writer.set_parameter[CistemParameters.position_in_stack](6)
            cistem_writer.set_parameter[CistemParameters.psi](4.3)
            cistem_writer.set_parameter[ExtraParameters.p_ind](7)
            cistem_writer.write_line()

        with LexicalScope():
            var cistem_writer = blocked_writer.write_block(
                block_second,
                num_lines = 3,
                cols = materialize[cols_second]()
            )

            cistem_writer.set_parameter[ExtraParameters.t_ind](5)
            cistem_writer.set_parameter[ExtraParameters.p_psi](3.4)
            cistem_writer.write_line()

            cistem_writer.set_parameter[ExtraParameters.t_ind](7)
            cistem_writer.set_parameter[ExtraParameters.p_psi](5.6)
            cistem_writer.write_line()

            cistem_writer.set_parameter[ExtraParameters.t_ind](9)
            cistem_writer.set_parameter[ExtraParameters.p_psi](7.8)
            cistem_writer.write_line()

    with LexicalScope():
        
        var reader = BytesReader(buf.span())
        var blocked_reader = BlockedReader(
            reader,
            blocks = [block_first, block_second],
            parameters = materialize[cols_first]() + materialize[cols_second]()
        )

        with LexicalScope():
            ref cistem_reader = blocked_reader.seek(block_first)

            assert_equal(len(cistem_reader.cols()), 3)
            assert_equal(cistem_reader.cols()[0], CistemParameters.position_in_stack)
            assert_equal(cistem_reader.cols()[1], CistemParameters.psi)
            assert_equal(cistem_reader.cols()[2], ExtraParameters.p_ind)
            assert_equal(cistem_reader.num_lines(), 2)

            cistem_reader.read_line()
            assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 5)
            assert_equal(cistem_reader.get_parameter[CistemParameters.psi](), 4.2)
            assert_equal(cistem_reader.get_parameter[ExtraParameters.p_ind](), 9)

            cistem_reader.read_line()
            assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 6)
            assert_equal(cistem_reader.get_parameter[CistemParameters.psi](), 4.3)
            assert_equal(cistem_reader.get_parameter[ExtraParameters.p_ind](), 7)

        with LexicalScope():
            ref cistem_reader = blocked_reader.seek(block_second)

            assert_equal(len(cistem_reader.cols()), 2)
            assert_equal(cistem_reader.cols()[0], ExtraParameters.t_ind)
            assert_equal(cistem_reader.cols()[1], ExtraParameters.p_psi)
            assert_equal(cistem_reader.num_lines(), 3)

            cistem_reader.read_line()
            assert_equal(cistem_reader.get_parameter[ExtraParameters.t_ind](), 5)
            assert_equal(cistem_reader.get_parameter[ExtraParameters.p_psi](), 3.4)

            cistem_reader.read_line()
            assert_equal(cistem_reader.get_parameter[ExtraParameters.t_ind](), 7)
            assert_equal(cistem_reader.get_parameter[ExtraParameters.p_psi](), 5.6)

            cistem_reader.read_line()
            assert_equal(cistem_reader.get_parameter[ExtraParameters.t_ind](), 9)
            assert_equal(cistem_reader.get_parameter[ExtraParameters.p_psi](), 7.8)
