
from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian, require_endian
from cryoluge.image import Image


# MRC file (from the Medical Research Council, in the UK)
# https://en.wikipedia.org/wiki/MRC_(file_format)

# format specification:
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php


alias _header_size = 1024

# only support little endian MRC files, for now
alias _endian = Endian.Little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var _reader: Pointer[R, origin]
    var _header: ByteBuffer

    fn __init__(
        out self,
        ref [origin] reader: R
    ) raises:
        self._reader = Pointer(to=reader)

        # the pixel read functions currently assume the native byte order matches the file byte order,
        # and we're only supporting little endian (for now),
        # so make sure the target architecture matches the code
        require_endian[_endian]()

        # read the MRC header into a buffer
        self._header = ByteBuffer(_header_size)
        self._reader[].read_bytes_exact(self._header.span())

        # read the machine stamp (at word 54)
        var r = BytesReader(self._header.span(), pos=_word_offset(54))
        var endian = MachineStamp(r.read_u8(), r.read_u8()).endian()
        if endian != _endian:
            raise Error("Reader supports only endian ", _endian, ", but file has endian ", endian)

    fn mode(self) raises -> Mode:

        # read the mode (at word 4)
        var r = BytesReader(self._header.span(), pos=_word_offset(4))
        var mode_value = r.read_u32[_endian]()
        return Mode.find(mode_value)

    fn size(self) raises -> (UInt32, UInt32, UInt32):
        var r = BytesReader(self._header.span(), pos=_word_offset(1))
        var nx = r.read_u32[_endian]()
        var ny = r.read_u32[_endian]()
        var nz = r.read_u32[_endian]()
        return (nx, ny, nz)

    fn _check_dtype[dtype: DType](self) raises:
        var mode = self.mode()
        if mode.dtype is None:
            raise Error(String("MRC file has mode ", mode, ", which is not supported by this function"))
        if mode.dtype.value() != dtype:
            raise Error(String("MRC file has mode ", mode, ", which expects DType ", mode.dtype.value(), ", but ", dtype, " was given instead"))

    fn _seek_pixels(self) raises:

        # read the extended header size (at word 24)
        var r = BytesReader(self._header.span(), pos=_word_offset(24))
        var extended_header_size = r.read_u32[_endian]()

        # seek to after the header and the extended header to get to the pixels
        self._reader[].seek_to(_header_size + UInt64(extended_header_size))

    fn read_3d[dtype: DType](self, out img: Image.D3[dtype]) raises:
        self._check_dtype[dtype]()
        var (nx, ny, nz) = self.size()
        img = Image.D3[dtype](sx=UInt(nx), sy=UInt(ny), sz=UInt(nz))
        self._seek_pixels()
        self._reader[].read_bytes_exact(img.span())

    fn read_3d_int8(self, out img: Image.D3[DType.int8]) raises:
        img = self.read_3d[DType.int8]()

    fn read_3d_float32(self, out img: Image.D3[DType.float32]) raises:
        img = self.read_3d[DType.float32]()

    # TODO: other supported dtypes

    fn read_2d[dtype: DType](self, z: UInt32, out img: Image.D2[dtype]) raises:
        self._check_dtype[dtype]()
        var (nx, ny, nz) = self.size()

        # check z
        if z >= nz:
            raise Error("z=", z, " is out of range [0,", nz, ")")

        img = Image.D2[dtype](sx=UInt(nx), sy=UInt(ny))
        self._seek_pixels()
        self._reader[].seek_by(Int64(nx)*Int64(ny)*Int64(z))
        self._reader[].read_bytes_exact(img.span())

    fn read_2d_int8(self, *, z: UInt32=0, out img: Image.D2[DType.int8]) raises:
        img = self.read_2d[DType.int8](z)

    fn read_2d_float32(self, *, z: UInt32=0, out img: Image.D2[DType.float32]) raises:
        img = self.read_2d[DType.float32](z)

    # TODO: other supported dtypes


fn _word_offset(word_number: Int) -> UInt:
    # words are numbered starting at 1 (sigh) and have 4 bytes each
    return (UInt(word_number) - 1)*4


fn _unsupported_mode[mode: Mode]():
    constrained[False, String("Mode not supported: ", mode)]()
