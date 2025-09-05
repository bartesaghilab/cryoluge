
from cryoluge.io import BinaryReader, Endian


alias endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
]:
    var reader: Pointer[R, origin]
    var parameters: ParameterSet
    var _cols: List[Column]
    var _line_size: Optional[Int]
    var _first_line_offset: UInt64

    fn __init__(
        out self,
        ref [origin] reader: R,
        *,
        parameters: Optional[ParameterSet] = None
    ) raises:
        self.reader = Pointer(to=reader)
        self.parameters = parameters.or_else(ParameterSet(CistemParameters.all))

        # read the number of columns and lines
        num_cols = self.reader[].read_i32[endian]()
        num_lines = self.reader[].read_i32[endian]()

        # read the column definitions
        self._cols = List[Column](capacity=Int(num_cols))
        for _ in range(num_cols):
            var id = self.reader[].read_i64[endian]()
            self._cols.append(Column(
                id,
                self.parameters[id].copied(),
                ColumnType.get(self.reader[].read_u8())
            ))

        # record the current file offset, for later line seeking
        self._first_line_offset = self.reader[].offset()

        # determine the constant line size, if any
        self._line_size: Optional[Int] = 0
        for col in self._cols:
            if col.type.size is None:
                self._line_size = None
                break
            else:
                self._line_size.value() += col.type.size.value()

    fn cols(self) -> ref [ImmutableOrigin.cast_from[__origin_of(self._cols)]] List[Column]:
        return self._cols

    fn seek_supported(self) -> Bool:
        return self._line_size is not None

    fn seek(self, linei: UInt) raises:

        # make sure seek is supported
        if not self.seek_supported():
            var variable_columns = [c for c in self._cols if c.type.size is None]
            raise Error(String("Seek not supported: Variable-width columns detected: ", ','.join(variable_columns)))
        
        # seek to the line offset
        self.reader[].seek_to(self._first_line_offset + linei*self._line_size.value())


@fieldwise_init
struct Column(Copyable, Movable, Writable):
    var id: Int64
    var parameter: Optional[Parameter]
    var type: ColumnType

    fn write_to[W: Writer](self, mut writer: W):
        if self.parameter:
            writer.write("Column[param=", self.parameter.value(), ", type=", self.type, "]")
        else:
            writer.write("Column[id=", self.id, ", type=", self.type, "]")            
