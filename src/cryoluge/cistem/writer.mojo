
from memory import memcpy

from cryoluge.io import BinaryWriter, BytesWriter
from cryoluge.lang import intcast_i32


comptime endian = Endian.Little


struct CistemWriter[
    W: BinaryWriter,
    origin: Origin[mut=True]
](Movable):
    var writer: Pointer[W, origin]
    var cols: List[Parameter]
    var _col_offsets: Dict[Int64,Int]
    var _line_buf: ByteBuffer

    fn __init__(
        out self,
        ref [origin] writer: W,
        *,
        num_lines: Int,
        var cols: List[Parameter] = CistemParameters.all()
    ) raises:
        self.writer = Pointer(to=writer)
        self.cols = cols^

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

    fn copy_line[
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
        
        memcpy(
            src=src.unsafe_ptr(),
            dest=dst.unsafe_ptr(),
            count=len(dst)
        )

    fn set_parameter[
        param: Parameter,
        dtype: DType
    ](
        mut self,
        v: Scalar[dtype]
    ) raises:

        # lookup the parameter offset, if any
        var offset = self._col_offsets.get(param.id)
        if offset is None:
            raise Error(String("No line offset for parameter: ", param))

        # mojo's compiler does type checking *before* function instantiation,
        # which means it's not "smart enough" to know that the parameter's dtype
        # matches the given dtype at type-checking time,
        # so explicitly check that the types match during function instantiation
        constrained[
            param.type.dtype.value() == dtype,
            String(
                "Expected parameter type with dtype=", dtype,
                ", but ", param.type.name, " has dtype=", param.type.dtype.value()
            )
        ]()

        var writer = BytesWriter(self._line_buf.span(), pos=offset.value())
        writer.write_scalar[dtype, endian](v)

    fn set_parameter[param: Parameter](
        mut self,
        v: IntLiteral
    ) raises:
        comptime dtype = param.type.dtype.value()
        self.set_parameter[param](Scalar[dtype](v))

    fn set_parameter[param: Parameter](
        mut self,
        v: FloatLiteral
    ) raises:
        comptime dtype = param.type.dtype.value()
        self.set_parameter[param](Scalar[dtype](v))

    fn write_line(mut self) raises:
        self.writer[].write_bytes(self._line_buf.span())

    fn transform[
        func: fn (CistemReader, mut CistemWriter) capturing raises -> Bool,
        R: BinaryReader,
        reader_origin: Origin[mut=True]
    ](mut self, mut reader: CistemReader[R,reader_origin], out num_written: Int) raises:
        num_written = 0
        while not reader.eof():
            reader.read_line()
            self.copy_line(reader)
            if func(reader, self):
                self.write_line()
                num_written += 1
