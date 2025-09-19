

alias endian = Endian.Little


struct BlockedReader[
    R: BinaryReader,
    origin: Origin[mut=True]
]:
    var reader: Pointer[R, origin]
    # TODO: need a collection type for non-copyable structs
    #var blocks: List[Reader[R, origin]]

    # TODO: nice init wrappers?

    fn __init__(
        out self,
        ref [origin] reader: R,
        var parameters: Dict[Int64,ParameterSet]  # indexed by block id
    ) raises:
        self.reader = Pointer(to=reader)
        # TODO: need a collection type for non-copyable structs
        #self.blocks = {}

        # read the block id
        var block_id = self.reader[].read_i64[endian]()

        # lookup the block parameters, if any
        var block_params = parameters.pop(block_id, default=ParameterSet([]))

        # read the block using a regular reader
        var block_reader = Reader(self.reader[], block_params^)

        # TODO: scan through the file to read the block ids and colums
        #var block_reader = Reader()
