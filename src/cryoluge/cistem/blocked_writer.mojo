
from cryoluge.io import Endian, BinaryWriter
from cryoluge.cistem.parameters import Parameter
from cryoluge.cistem.blocks import Block
from cryoluge.cistem.writer import CistemWriter


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
    var _writer: Pointer[Self.W, Self.origin]

    def __init__(
        out self,
        ref [Self.origin] writer: Self.W
    ):
        self._writer = Pointer(to=writer)

    def write_block[Parameters: Iterable, //](
        mut self,
        block: Block,
        *,
        num_lines: Int,
        parameters: Parameters
    ) raises -> CistemWriter[Self.W, Self.origin]
        where Parameters.IteratorType[origin_of(parameters)].Element == Parameter:

        # write the block id
        self._writer[].write_i64[endian](block.id)

        # write the block header
        return CistemWriter(
            self._writer[],
            parameters = parameters,
            num_lines = num_lines,
        )
