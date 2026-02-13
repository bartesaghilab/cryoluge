
from cryoluge.collections import MovableList, Keyable


comptime endian = Endian.Little


struct BlockedReader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var _reader: Pointer[R, origin]
    var _blocks: KeyableSet[Block]
    var _parameters: KeyableSet[Parameter]
    var _blocks_present: List[Block]
    var _block_reader_indices: Dict[Block.Key, Int]
    var _block_readers: MovableList[CistemReader[R, origin]]

    fn __init__(
        out self,
        ref [origin] reader: R
    ) raises:
        self = Self(reader, blocks=[], parameters=CistemParameters.all())

    # TODO: can we use Iterable trait here?
    fn __init__(
        out self,
        ref [origin] reader: R,
        *,
        blocks: List[Block],
        parameters: List[Parameter]
    ) raises:
        self = Self(reader, KeyableSet[Block](blocks), KeyableSet[Parameter](parameters))

    fn __init__(
        out self,
        ref [origin] reader: R,
        var blocks: KeyableSet[Block],
        var parameters: KeyableSet[Parameter]
    ) raises:
        self._reader = Pointer(to=reader)
        self._blocks = blocks^
        self._parameters = parameters^
        self._blocks_present = []
        self._block_reader_indices = {}
        self._block_readers = MovableList[CistemReader[R, origin]]()

        while True:

            # read the block id, if any
            var block_id: Int64
            try:
                block_id = self._reader[].read_i64[endian]()
            except ex:
                break

            # look up the block
            var block: Block
            var maybe_block = self._blocks[block_id]
            if maybe_block is not None:
                block = maybe_block.value()[]
            else:
                block = Block.unknown(block_id)
            self._blocks_present.append(block)

            # read the cistem metadata (but not the lines) from the block
            var block_reader = CistemReader(self._reader[], self._parameters.copy())

            # skip to the end of the block, so we can read the next one
            block_reader.seek(block_reader.num_lines())

            # save the block reader for later
            self._block_reader_indices[block.id] = len(self._block_readers)
            self._block_readers.append(block_reader^)

    fn blocks(ref self) -> ref [origin_of(self._blocks_present)] List[Block]:
        return self._blocks_present

    fn seek(mut self, block: Block) raises -> ref [self._block_readers] CistemReader[R, origin]:

        # lookup the block
        var maybe_i = self._block_reader_indices.get(block.id)
        if maybe_i is None:
            raise Error(String('Block ', block, ' was not found'))
        var i = maybe_i.value()

        ref block_reader = self._block_readers[i]

        # seek to the start of the block lines
        block_reader.seek(0)

        return block_reader
