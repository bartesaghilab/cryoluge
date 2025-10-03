
from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian


# MRC file (from the Medical Research Council, in the UK)
# https://en.wikipedia.org/wiki/MRC_(file_format)

# format specification:
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php


alias MachineStamp = InlineArray[Byte, 2]


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var reader: Pointer[R, origin]
    var header: ByteBuffer

    alias _machine_stamp_little = MachineStamp(0x44, 0x44)
    alias _machine_stamp_big = MachineStamp(0x11, 0x11)

    fn __init__(
        out self,
        ref [origin] reader: R
    ) raises:
        self.reader = Pointer(to=reader)

        # Read the MRC header into a buffer.
        # The header is always 1024 bytes.
        self.header = ByteBuffer(1024)
        self.reader[].read_bytes_exact(self.header.span())

    fn endian(self) raises -> Endian:
        # the machine stamp is at word 54
        var r = BytesReader(self.header.span(), pos=_word_offset(54))
        var machine_stamp = InlineArray[Byte, 2](
            r.read_u8(),
            r.read_u8()
        )
        if machine_stamp_eq(machine_stamp, Self._machine_stamp_little):
            return Endian.Little
        elif machine_stamp_eq(machine_stamp, Self._machine_stamp_big):
            return Endian.Big
        else:
            raise Error(String("Can't determine endianness: machine stamp unrecognized: ", machine_stamp_to_string(machine_stamp)))


fn _word_offset(word_number: Int) -> UInt:
    # words are numbered starting at 1 (sigh) and have 4 bytes each
    return (UInt(word_number) - 1)*4


fn machine_stamp_eq(obs: MachineStamp, exp: MachineStamp) -> Bool:
    return obs[0] == exp[0] and obs[1] == exp[1]


fn machine_stamp_to_string(stamp: MachineStamp) -> String:
    return String("[", hex(stamp[0]), ", ", hex(stamp[1]), "]")
