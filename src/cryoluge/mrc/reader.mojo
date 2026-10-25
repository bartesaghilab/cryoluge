
from std.sys import size_of, Endian

from cryoluge.io import BinaryReader, ByteBuffer, BytesReader, Endian, require_endian, OwnedFileReader
from cryoluge.image import Image, Vec
from cryoluge.mrc.mode import Mode
from cryoluge.mrc.machine_stamp import MachineStamp


# MRC file (from the Medical Research Council, in the UK)
# https://en.wikipedia.org/wiki/MRC_(file_format)

# format specification:
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php


comptime _header_size = 1024

# only support little endian MRC files, for now
comptime _endian = Endian.little


struct Reader[
    R: BinaryReader,
    origin: Origin[mut=True]
](Movable):
    var _reader: Pointer[Self.R, Self.origin]
    var _inner: _Inner

    def __init__(
        out self,
        ref [Self.origin] reader: Self.R
    ) raises:
        self._reader = Pointer(to=reader)
        self._inner = _Inner(self._reader[])

    def mode(self) raises -> Mode:
        return self._inner.mode()

    def size(self) raises -> Tuple[UInt32, UInt32, UInt32]:
        return self._inner.size()

    def size_3(self) raises -> Vec[3, .int]:
        return self._inner.size_3()

    def size_2(self) raises -> Vec[2, .int]:
        return self._inner.size_2()

    def read_3d[dtype: DType](self, mut img: Image[3,dtype]) raises:
        return self._inner.read_3d(self._reader[], img)

    def read_3d_int8(self, mut img: Image[3,DType.int8]) raises:
        self.read_3d[DType.int8](img)

    def read_3d_float32(self, mut img: Image[3,DType.float32]) raises:
        self.read_3d[DType.float32](img)

    # TODO: other supported dtypes

    def read_2d[dtype: DType](self, mut img: Image[2,dtype], *, z: UInt32) raises:
        return self._inner.read_2d(self._reader[], img, z=z)

    def read_2d_int8(self, mut img: Image[2,DType.int8], *, z: UInt32=0) raises:
        self.read_2d[DType.int8](img, z=z)

    def read_2d_float32(self, mut img: Image[2,DType.float32], *, z: UInt32=0) raises:
        self.read_2d[DType.float32](img, z=z)

    # TODO: other supported dtypes


struct OwnedReader[
    R: BinaryReader & Movable & Deinitable = OwnedFileReader
](Movable):
    var _reader: Self.R
    var _inner: _Inner

    def __init__(
        out self,
        var reader: Self.R
    ) raises:
        self._reader = reader^
        self._inner = _Inner(self._reader)

    @staticmethod
    def open(path: String, out reader: OwnedReader[OwnedFileReader]) raises:
        reader = OwnedReader(OwnedFileReader(path))

    def mode(self) raises -> Mode:
        return self._inner.mode()

    def size(self) raises -> Tuple[UInt32, UInt32, UInt32]:
        return self._inner.size()

    def size_3(self) raises -> Vec[3, .int]:
        return self._inner.size_3()

    def size_2(self) raises -> Vec[2, .int]:
        return self._inner.size_2()

    def read_3d[dtype: DType](mut self, mut img: Image[3,dtype]) raises:
        return self._inner.read_3d(self._reader, img)

    def read_3d_int8(mut self, mut img: Image[3,DType.int8]) raises:
        self.read_3d[DType.int8](img)

    def read_3d_float32(mut self, mut img: Image[3,DType.float32]) raises:
        self.read_3d[DType.float32](img)

    # TODO: other supported dtypes

    def read_2d[dtype: DType](mut self, mut img: Image[2,dtype], *, z: UInt32) raises:
        return self._inner.read_2d(self._reader, img, z=z)

    def read_2d_int8(mut self, mut img: Image[2,DType.int8], *, z: UInt32=0) raises:
        self.read_2d[DType.int8](img, z=z)

    def read_2d_float32(mut self, mut img: Image[2,DType.float32], *, z: UInt32=0) raises:
        self.read_2d[DType.float32](img, z=z)

    # TODO: other supported dtypes


struct _Inner(Movable):

    var _header: ByteBuffer

    def __init__[R: BinaryReader](out self, mut reader: R) raises:

        # the pixel read functions currently assume the native byte order matches the file byte order,
        # and we're only supporting little endian (for now),
        # so make sure the target architecture matches the code
        require_endian[_endian]()

        # read the MRC header into a buffer
        self._header = ByteBuffer(_header_size)
        reader.read_bytes_exact(self._header.span())

        # read the machine stamp (at word 54)
        var r = BytesReader(self._header.span(), pos=_word_offset(54))
        var endian = MachineStamp(r.read_u8(), r.read_u8()).endian()
        if endian != _endian:
            raise Error("Reader supports only endian ", _endian, ", but file has endian ", endian)

    def mode(self) raises -> Mode:

        # read the mode (at word 4)
        var r = BytesReader(self._header.span(), pos=_word_offset(4))
        var mode_value = r.read_u32[_endian]()
        return Mode.find(mode_value)

    def size(self) raises -> Tuple[UInt32, UInt32, UInt32]:
        var r = BytesReader(self._header.span(), pos=_word_offset(1))
        var nx = r.read_u32[_endian]()
        var ny = r.read_u32[_endian]()
        var nz = r.read_u32[_endian]()
        return (nx, ny, nz)

    def size_3(self) raises -> Vec[3, .int]:
        var (sx, sy, sz) = self.size()
        return Vec[3](x=Int(sx), y=Int(sy), z=Int(sz))

    def size_2(self) raises -> Vec[2, .int]:
        var (sx, sy, _) = self.size()
        return Vec[2](x=Int(sx), y=Int(sy))

    def _check_dtype[dtype: DType](self) raises:
        var mode = self.mode()
        if mode.dtype is None:
            raise Error(String("MRC file has mode ", mode, ", which is not supported by this function"))
        if mode.dtype.value() != dtype:
            raise Error(String("MRC file has mode ", mode, ", which expects DType ", mode.dtype.value(), ", but ", dtype, " was given instead"))

    def _seek_pixels[R: BinaryReader](self, mut reader: R) raises:

        # read the extended header size (at word 24)
        var r = BytesReader(self._header.span(), pos=_word_offset(24))
        var extended_header_size = r.read_u32[_endian]()

        # seek to after the header and the extended header to get to the pixels
        reader.seek_to(_header_size + Int(extended_header_size))

    def read_3d[dtype: DType, R: BinaryReader](self, mut reader: R, mut img: Image[3,dtype]) raises:
        self._check_dtype[dtype]()
        var size = self.size_3()
        if img.sizes() != size:
            raise Error("Image should have size ", size, ", but instead it has size ", img.sizes())
        self._seek_pixels(reader)
        reader.read_bytes_exact(img.span_bytes())

    def read_2d[dtype: DType, R: BinaryReader](self, mut reader: R, mut img: Image[2,dtype], *, z: UInt32) raises:
        self._check_dtype[dtype]()
        var (_, _, sz) = self.size()
        var size = self.size_2()
        if img.sizes() != size:
            raise Error("Image should have size ", size, ", but instead it has size ", img.sizes())

        # check z
        if z >= sz:
            raise Error("z=", z, " is out of range [0,", sz, ")")

        self._seek_pixels(reader)
        reader.seek_by(size.x()*size.y()*Int(z)*size_of[dtype]())
        reader.read_bytes_exact(img.span_bytes())


def _word_offset(word_number: Int) -> Int:
    # words are numbered starting at 1 (sigh) and have 4 bytes each
    return (word_number - 1)*4


def _unsupported_mode[mode: Mode]() -> Never:
    comptime assert False, String("Mode not supported: ", mode)
