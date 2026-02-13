
comptime endian = Endian.Little


@fieldwise_init
struct BlockWriteInfo(Copyable, Movable):
    var block: Block
    var cols: List[Parameter]
    var num_lines: Int


struct BlockedWriter[
    W: BinaryWriter,
    origin: Origin[mut=True]
](Movable):
    var _writer: Pointer[W, origin]

    fn __init__(
        out self,
        ref [origin] writer: W
    ):
        self._writer = Pointer(to=writer)

    fn write_block(
        mut self,
        block: Block,
        *,
        num_lines: Int,
        var cols: List[Parameter]
    ) raises -> CistemWriter[W, origin]:

        # write the block id
        self._writer[].write_i64[endian](block.id)

        # write the block header
        return CistemWriter(
            self._writer[],
            num_lines = num_lines,
            cols = cols^
        )
