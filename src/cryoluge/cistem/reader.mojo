
from cryoluge.io import BinaryReader, Endian


alias endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
]:
    var reader: Pointer[R, origin]
    var parameters: ParameterSet
    var _cols: List[Column]
    var _first_line_offset: UInt64
    var _line_size: UInt
    var _col_offsets: Dict[Int64,UInt]

    fn __init__(
        out self,
        ref [origin] reader: R,
        *,
        parameters: Optional[ParameterSet] = None
    ) raises:
        self.reader = Pointer(to=reader)
        self.parameters = parameters.or_else(ParameterSet(CistemParameters.all()))

        # read the number of columns and lines
        var num_cols = self.reader[].read_i32[endian]()
        var num_lines = self.reader[].read_i32[endian]()

        # read the column definitions
        self._cols = List[Column](capacity=Int(num_cols))
        for _ in range(num_cols):
            var id = self.reader[].read_i64[endian]()
            self._cols.append(Column(
                id,
                self.parameters[id].copied(),
                ColumnType.get(self.reader[].read_u8())
            ))

        # check for variable-width columns, which aren't supported here
        var variable_columns = [c for c in self._cols if c.type.size is None]
        if len(variable_columns) > 0:
            raise Error(String("Variable-width columns not supported: ", ','.join(variable_columns)))

        # record the current file offset, for later line seeking
        self._first_line_offset = self.reader[].offset()

        # compute the line size and column offsets
        self._line_size = 0
        self._col_offsets = {}
        for col in self._cols:
            self._col_offsets[col.id] = self._line_size
            self._line_size += col.type.size.value()

    fn cols(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._cols)]] List[Column]:
        return self._cols

    fn seek(self, linei: UInt) raises:
        self.reader[].seek_to(self._first_line_offset + linei*self._line_size)
        # TODO: ensure the line is buffered

    fn read_col[col: Column](self) raises:
        # TODO
        pass


@fieldwise_init
struct Column(ImplicitlyCopyable, Movable, Writable):
    var id: Int64
    var parameter: Optional[Parameter]
    var type: ColumnType

    fn write_to[W: Writer](self, mut writer: W):
        if self.parameter:
            writer.write("Column[param=", self.parameter.value(), ", type=", self.type, "]")
        else:
            writer.write("Column[id=", self.id, ", type=", self.type, "]")            
