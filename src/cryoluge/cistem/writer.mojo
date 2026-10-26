
from std.sys import Endian

from cryoluge.io import ByteBuffer, BinaryWriter, BytesWriter, BinaryReader, Endian
from cryoluge.lang import intcast_i32
from cryoluge.cistem.parameters import Parameter, CistemParameters
from cryoluge.cistem.reader import CistemReader


comptime endian = Endian.little


struct CistemWriter[
    W: BinaryWriter,
    origin: Origin[mut=True]
](Movable):
    var writer: Pointer[Self.W, Self.origin]
    var cols: List[Parameter]
    var _col_offsets: Dict[Int64,Int]
    var _line_buf: ByteBuffer

    def __init__(
        out self,
        ref [Self.origin] writer: Self.W,
        *,
        num_lines: Int
    ) raises:
        self = Self(writer, parameters=CistemParameters.all(), num_lines=num_lines)

    def __init__(
        out self,
        ref [Self.origin] writer: Self.W,
        *,
        parameters: Some[Iterable],
        num_lines: Int
    ) raises:

        self.writer = Pointer(to=writer)

        self.cols = List[Parameter]()
        for var p in parameters:
            self.cols.append(rebind_var[Parameter](p^))

        # check for variable-width columns, which aren't supported here
        var variable_columns = [c for c in self.cols if c.type.dtype is None]
        if len(variable_columns) > 0:
            raise Error(String("Variable-width columns not supported: ", ','.join(variable_columns)))

        # compute the line size and column offsets
        var line_size = 0
        self._col_offsets = Dict[Int64,Int]()
        for col in self.cols:
            self._col_offsets[col.id] = line_size
            line_size += col.type.size.value()
        self._line_buf = ByteBuffer(line_size)

        # write the number of columns and lines
        var num_cols = len(self.cols)
        var num_cols_i32: Int32
        try:
            num_cols_i32 = intcast_i32(num_cols)
        except err:
            raise Error("Too many parameters: ", err)
        self.writer[].write_i32[endian](num_cols_i32)
        var num_lines_i32: Int32
        try:
            num_lines_i32 = intcast_i32(num_lines)
        except err:
            raise Error("Too many lines: ", err)
        self.writer[].write_i32[endian](num_lines_i32)

        # write the column definitions
        for col in self.cols:
            self.writer[].write_i64[endian](col.id)
            self.writer[].write_u8(col.type.id)

    def copy_line[
        R: BinaryReader,
        origin_reader: Origin[mut=True]
    ](
        mut self,
        reader: CistemReader[R, origin_reader]
    ) raises:

        var src = reader._line_buf.span()
        var dst = self._line_buf.span()

        if len(src) != len(dst):
            raise Error("Can't copy line buffer, sizes are different: ", len(src), " -> ", len(dst))
        
        dst.copy_from(src)

    def set_parameter[
        dtype: DType,
        //,
        param: Parameter
    ](
        mut self,
        v: Scalar[dtype]
    ) raises:

        # constraints can't see to handle this check, so just make a compile-time assert instead
        comptime assert param.type.dtype.value() == dtype,
            String(
                "Expected parameter type with dtype=", dtype,
                ", but ", param.type.name, " has dtype=", param.type.dtype.value()
            )

        # lookup the parameter offset, if any
        var offset = self._col_offsets.get(param.id)
        if offset is None:
            raise Error(String("No line offset for parameter: ", param))

        var writer = BytesWriter(self._line_buf.span(), pos=offset.value())
        writer.write_scalar[dtype, endian](v)

    def set_parameter[param: Parameter](
        mut self,
        v: IntLiteral
    ) raises:
        comptime dtype = param.type.dtype.value()
        self.set_parameter[param](Scalar[dtype](v))

    def set_parameter[param: Parameter](
        mut self,
        v: FloatLiteral
    ) raises:
        comptime dtype = param.type.dtype.value()
        self.set_parameter[param](Scalar[dtype](v))

    def write_line(mut self) raises:
        self.writer[].write_bytes(self._line_buf.span())

    def transform[
        R: BinaryReader,
        reader_origin: Origin[mut=True]
    ](
        mut self,
        mut reader: CistemReader[R,reader_origin],
        transformer: Some[def (CistemReader, mut CistemWriter) raises -> Bool],
        out num_written: Int
    ) raises:
        num_written = 0
        while not reader.eof():
            reader.read_line()
            self.copy_line(reader)
            if transformer(reader, self):
                self.write_line()
                num_written += 1
