
from cryoluge.collections import KeyableSet
from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian


alias endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var reader: Pointer[R, origin]
    var parameters: KeyableSet[Parameter]
    var _num_lines: UInt
    var _cols: List[Parameter]
    var _first_line_offset: UInt64
    var _col_offsets: Dict[Int64,UInt]
    var _line_buf: ByteBuffer
    var _next_line: UInt

    fn __init__(
        out self,
        ref [origin] reader: R
    ) raises:
        self = Self(reader, parameters=CistemParameters.all())


    # TODO: can we use Iterable trait here?
    fn __init__(
        out self,
        ref [origin] reader: R,
        *,
        parameters: List[Parameter]
    ) raises:
        self = Self(reader, KeyableSet[Parameter](parameters))

    fn __init__(
        out self,
        ref [origin] reader: R,
        var parameters: KeyableSet[Parameter]
    ) raises:
        self.reader = Pointer(to=reader)
        self.parameters = parameters^

        # read the number of columns and lines
        var num_cols = self.reader[].read_i32[endian]()
        var num_lines = self.reader[].read_i32[endian]()
        if num_lines < 0:
            raise Error(String("Invalid number of lines: ", num_lines))
        self._num_lines = UInt(num_lines)

        # read the column definitions
        self._cols = List[Parameter](capacity=Int(num_cols))
        for coli in range(num_cols):

            # read the raw ids
            var id = self.reader[].read_i64[endian]()
            var type_id = self.reader[].read_u8()

            # try to lookup the type
            var _type = ParameterType.get(type_id)
            if _type is None:
                raise Error(String("Column index=", coli, ",id=", id, " has unrecognized type_id=", type_id)) 
            var type = _type.value()

            # try to look up the parameter
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
        self._next_line = 0

        # compute the line size and column offsets
        var line_size = UInt(0)
        self._col_offsets = Dict[Int64,UInt]()
        for col in self._cols:
            self._col_offsets[col.id] = line_size
            line_size += col.type.dtype.value().size_of()
        self._line_buf = ByteBuffer(line_size)

    fn num_lines(self) -> UInt:
        return self._num_lines

    fn line_size(self) -> UInt:
        return self._line_buf.size()

    fn cols(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._cols)]] List[Parameter]:
        return self._cols

    fn has_parameter[param: Parameter](self) -> Bool:
        return self._col_offsets.get(param.id) is not None

    fn seek(mut self, linei: UInt) raises:
        self.reader[].seek_to(self._first_line_offset + UInt64(linei*self._line_buf.size()))
        self._next_line = linei

    fn next_line(self) -> UInt:
        return self._next_line

    fn eof(self) -> Bool:
        return self._next_line >= self._num_lines

    fn read_line(mut self) raises:
        self.reader[].read_bytes_exact(self._line_buf.span())
        # TODO: can we get rid of this copy? ^^
        #       need some kind of window function on the BytesReader
        self._next_line += 1
   
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
