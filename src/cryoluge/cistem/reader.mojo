
from cryoluge.io import BinaryReader, Endian


alias endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
]:
    var reader: Pointer[R, origin]

    fn __init__(out self, ref [origin] reader: R) raises:
        self.reader = Pointer(to=reader)

        # read the number of columns and lines
        num_cols = self.reader[].read_i32[endian]()
        num_lines = self.reader[].read_i32[endian]()

        print('cols=', num_cols, ', lines=', num_lines)  # TEMP

        # TODO: NEXTTIME: keep reading!!
