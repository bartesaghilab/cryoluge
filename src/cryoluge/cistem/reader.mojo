
from cryoluge.lang import rebind_scalar
from cryoluge.collections import KeyableSet
from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian


comptime endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var reader: Pointer[R, origin]
    var parameters: KeyableSet[Parameter]
    var _num_lines: Int
    var _cols: List[Parameter]
    var _first_line_offset: UInt64
    var _col_offsets: Dict[Int64,Int]
    var _line_buf: ByteBuffer
    var _next_line: Int

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
        self._num_lines = Int(num_lines)

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
        var line_size = 0
        self._col_offsets = Dict[Int64,Int]()
        for col in self._cols:
            self._col_offsets[col.id] = line_size
            line_size += col.type.size.value()
        self._line_buf = ByteBuffer(line_size)

    fn num_lines(self) -> Int:
        return self._num_lines

    fn line_size(self) -> Int:
        return self._line_buf.size()

    fn cols(self) -> ref [origin_of(self._cols)] List[Parameter]:
        return self._cols

    fn has_parameter[param: Parameter](self) -> Bool:
        return self._col_offsets.get(param.id) is not None

    fn seek(mut self, linei: Int) raises:
        self.reader[].seek_to(self._first_line_offset + UInt64(linei*self._line_buf.size()))
        self._next_line = linei

    fn next_line(self) -> Int:
        return self._next_line

    fn eof(self) -> Bool:
        return self._next_line >= self._num_lines

    fn read_line(mut self) raises:
        self.reader[].read_bytes_exact(self._line_buf.span())
        # TODO: can we get rid of this copy? ^^
        #       need some kind of window function on the BytesReader
        self._next_line += 1
   
    fn get_parameter[
        param: Parameter,
        dtype: DType=param.type.dtype.value()
    ](self, out v: Scalar[dtype]) raises:

        # lookup the parameter offset, if any
        var offset = self._col_offsets.get(param.id)
        if offset is None:
            raise Error(String("No line offset for parameter: ", param))

        var reader = BytesReader(self._line_buf.span(), pos=offset.value())
        # TODO: any way to store the reader in the struct?
        #       how does self-referential struct syntax work?
        #       is that even faster than this? can the compiler optimize enough? need to profile

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
        v = reader.read_scalar[dtype, endian]()

    fn _get_parameter[
        param: Parameter,
        type: ParameterType,
        dtype: DType
    ](self, out v: Scalar[dtype]) raises:
        constrained[
            param.type == type,
            String("Expected parameter with ", type.name, " type, but ", param.name, " has ", param.type.name, " type")
        ]()
        var param_value = self.get_parameter[param, dtype]()
        v = rebind_scalar[dtype](param_value)

    # tragically, mojo's compiler isn't "smart enough" to use parametric return types
    # with the ergonomics you'd expect
    # when the return type parameter is the result of a compile-time expression,
    # so we still need to make type-specific wrapper functions,
    # but at least they can be very simple wrappers =)

    fn get_parameter_bool[param: Parameter](self, out v: Bool) raises:
        v = self._get_parameter[param, ParameterType.bool, DType.bool]()

    fn get_parameter_byte[param: Parameter](self, out v: Byte) raises:
        v = self._get_parameter[param, ParameterType.byte, DType.uint8]()

    fn get_parameter_int[param: Parameter](self, out v: Int32) raises:
        v = self._get_parameter[param, ParameterType.int, DType.int32]()

    fn get_parameter_uint[param: Parameter](self, out v: UInt32) raises:
        v = self._get_parameter[param, ParameterType.uint, DType.uint32]()

    fn get_parameter_long[param: Parameter](self, out v: Int64) raises:
        v = self._get_parameter[param, ParameterType.long, DType.int64]()

    fn get_parameter_float[param: Parameter](self, out v: Float32) raises:
        v = self._get_parameter[param, ParameterType.float, DType.float32]()

    fn get_parameter_double[param: Parameter](self, out v: Float64) raises:
        v = self._get_parameter[param, ParameterType.double, DType.float64]()

    fn get_parameter_into_string(self, param: Parameter) raises -> String:

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

        # parameter doesn't have a dtype
        raise Error(String("Failed to match scalar type to parameter: ", param))

    fn line_string(self) raises -> String:
        var out = List[String]()
        for col in self._cols:
            out.append(String(col, '=', self.get_parameter_into_string(col)))
        return ', '.join(out)
