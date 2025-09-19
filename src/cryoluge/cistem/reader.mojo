
from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian


alias endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
]:
    var reader: Pointer[R, origin]
    var parameters: ParameterSet
    var _num_lines: Int32
    var _cols: List[Parameter]
    var _first_line_offset: UInt64
    var _col_offsets: Dict[Int64,UInt]
    var _line_buf: ByteBuffer

    fn __init__(
        out self,
        ref [origin] reader: R
    ) raises:
        self = Self(reader, parameters=CistemParameters.all())


    # TODO: can we use Iterator trait here?
    fn __init__(
        out self,
        ref [origin] reader: R,
        *,
        parameters: List[Parameter]
    ) raises:
        self = Self(reader, ParameterSet(parameters))

    fn __init__(
        out self,
        ref [origin] reader: R,
        var parameters: ParameterSet
    ) raises:
        self.reader = Pointer(to=reader)
        self.parameters = parameters^

        # read the number of columns and lines
        var num_cols = self.reader[].read_i32[endian]()
        self._num_lines = self.reader[].read_i32[endian]()

        # read the column definitions
        self._cols = List[Parameter](capacity=Int(num_cols))
        for _ in range(num_cols):
            var id = self.reader[].read_i64[endian]()
            var type = ParameterType.get(self.reader[].read_u8())
            param = self.parameters[id]
            if param is not None:
                var p = param.value()[]

                # known parameter: check the type
                if param.value()[].type != type:
                    raise Error(String("Parameter '", p.name, "' expects type ", p.type, ", but file has type ", type))

                self._cols.append(p)

            else:
                # unknown parameter
                self._cols.append(Parameter.unknown(id, type))

        # check for variable-width columns, which aren't supported here
        var variable_columns = [c for c in self._cols if c.type.dtype is None]
        if len(variable_columns) > 0:
            raise Error(String("Variable-width columns not supported: ", ','.join(variable_columns)))

        # record the current file offset, for later line seeking
        self._first_line_offset = self.reader[].offset()

        # compute the line size and column offsets
        var line_size = UInt(0)
        self._col_offsets = Dict[Int64,UInt]()
        for col in self._cols:
            self._col_offsets[col.id] = line_size
            line_size += col.type.dtype.value().size_of()
        self._line_buf = ByteBuffer(line_size)

    fn num_lines(self) -> Int32:
        return self._num_lines

    fn cols(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._cols)]] List[Parameter]:
        return self._cols

    fn has_parameter[param: Parameter](self) -> Bool:
        return self._col_offsets.get(param.id) is not None

    fn seek(self, linei: UInt) raises:
        self.reader[].seek_to(self._first_line_offset + linei*self._line_buf.size())

    fn read_line(mut self) raises:
        self.reader[].read_bytes_exact(self._line_buf.span())
        # TODO: can we get rid of this copy? ^^
        #       need some kind of window function on the BytesReader
   
    fn get_parameter[param: Parameter](self, out v: Scalar[param.type.dtype.value()]) raises:

        # lookup the parameter offset, if any
        var offset = self._col_offsets.get(param.id)
        if offset is None:
            raise Error(String("No line offset for parameter: ", param))

        var reader = BytesReader(self._line_buf.span(), pos=offset.value())
        # TODO: any way to store the reader in the struct?
        #       how does self-referential struct syntax work?
        #       is that even faster than this? can the compiler optimize enough? need to profile
        v = reader.read_scalar[param.type.dtype.value(), endian]()

    fn get_parameter_string(self, param: Parameter) raises -> String:

        # lookup the parameter offset, if any
        var offset = self._col_offsets.get(param.id)
        if offset is None:
            raise Error(String("No line offset for parameter: ", param))
        
        var reader = BytesReader(self._line_buf.span(), pos=offset.value())
        # TODO: any way to store the reader in the struct?

        @parameter
        for type in ParameterType.all:
            @parameter
            if type.dtype is not None:
                if type == param.type:
                    var v = reader.read_scalar[type.dtype.value(), endian]()
                    return String(v)

        # shouldn't be possible to get here, but just in case
        raise Error(String("Failed to match scalar type to parameter: ", param))

    fn line_string(self) raises -> String:
        var out = List[String]()
        for col in self._cols:
            out.append(String(col, '=', self.get_parameter_string(col)))
        return ', '.join(out)
