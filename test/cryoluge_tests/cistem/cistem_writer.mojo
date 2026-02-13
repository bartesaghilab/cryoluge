
from testing import assert_equal

from cryoluge.lang import LexicalScope
from cryoluge.io import ByteBuffer, BytesWriter, BytesReader
from cryoluge.cistem import CistemWriter, CistemReader, CistemParameters


comptime funcs = __functions_in_module()


def test_write_lines():

    var params = [
        CistemParameters.position_in_stack,  # uint => UInt32 => 4
        CistemParameters.psi,  # float => Float32 => 4
        ExtraParameters.p_ind,  # int => Int32 => 4
    ]

    var buf = ByteBuffer(59)  # 35 + 2*12
    # header: 4 + 4 + 3*(8+1) = 35
    # line: 4 + 4 + 4 = 12

    with LexicalScope():
        var writer = BytesWriter(buf.span())
        var cistem_writer = CistemWriter(writer, num_lines=2, cols=params.copy())

        cistem_writer.set_parameter[CistemParameters.position_in_stack](5)
        cistem_writer.set_parameter[CistemParameters.psi](4.2)
        cistem_writer.set_parameter[ExtraParameters.p_ind](9)
        cistem_writer.write_line()

        cistem_writer.set_parameter[CistemParameters.position_in_stack](UInt32(6))
        cistem_writer.set_parameter[CistemParameters.psi](Float32(4.3))
        cistem_writer.set_parameter[ExtraParameters.p_ind](Int32(7))
        cistem_writer.write_line()

    with LexicalScope():
        var reader = BytesReader(buf.span())
        var cistem_reader = CistemReader(reader, parameters=params)

        # check the cols
        assert_equal(len(cistem_reader.cols()), 3)
        assert_equal(cistem_reader.cols()[0], CistemParameters.position_in_stack)
        assert_equal(cistem_reader.cols()[1], CistemParameters.psi)
        assert_equal(cistem_reader.cols()[2], ExtraParameters.p_ind)

        # check the lines
        assert_equal(cistem_reader.num_lines(), 2)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 5)
        assert_equal(cistem_reader.get_parameter[CistemParameters.psi](), 4.2)
        assert_equal(cistem_reader.get_parameter[ExtraParameters.p_ind](), 9)

        cistem_reader.read_line()
        assert_equal(cistem_reader.get_parameter[CistemParameters.position_in_stack](), 6)
        assert_equal(cistem_reader.get_parameter[CistemParameters.psi](), 4.3)
        assert_equal(cistem_reader.get_parameter[ExtraParameters.p_ind](), 7)
