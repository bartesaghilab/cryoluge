
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from sys.info import alignof, sizeof
from bit import byte_swap


trait BinaryReader:
    fn read_bytes(mut self, buf: Span[mut=True, Byte, alignment=alignof[Byte]()]) raises -> UInt: ...
    # For some reason, we need to explicitly specify Byte alignment
    # even though the default alignment is supposed to be `alignof[T]()`.
    # Tragically, this means the explicit alignment must infect all implementors of this trait.


fn _as_mut_byte_span[
    dtype: DType,
    origin: Origin[mut=True]
](ref [origin] v: Scalar[dtype]) -> Span[
    mut=True,
    Byte,
    origin,
    alignment=alignof[Byte]()
]:
    return Span(
        UnsafePointer(to=v)
            .bitcast[Byte]()
            .static_alignment_cast[alignof[Byte]()](),
        dtype.sizeof()
    )


fn _check_read[T: AnyType](bytes_read: UInt) raises:
    var size = sizeof[T]()
    if bytes_read != size:
        raise Error("Underflow: read ", bytes_read, ", of ", size, " byte(s)")


# NOTE: can't use stdlib's swap() for this because of aliasing rules =(
fn _swap_in_span(s: Span[mut=True, Byte, alignment=alignof[Byte]()], i1: UInt, i2: UInt):
    debug_assert(i1 < len(s))
    debug_assert(i2 < len(s))
    var swap = s[i1]
    s[i1] = s[i2]
    s[i2] = swap


fn _swap_bytes_if_needed[dtype: DType, endian: Endian](mut v: Scalar[dtype]):
    @parameter
    if endian != Endian.native():
        @parameter
        if dtype.is_integral():
            v = byte_swap(v)
        else:
            # no fancy intrinsic fn for non-int types, so just do it the hard way
            var s = _as_mut_byte_span(v)
            @parameter
            if dtype.sizeof() == 4:
                _swap_in_span(s, 0, 3)
                _swap_in_span(s, 1, 2)
            elif dtype.sizeof() == 8:
                _swap_in_span(s, 0, 7)
                _swap_in_span(s, 1, 6)
                _swap_in_span(s, 2, 5)
                _swap_in_span(s, 3, 4)
            else:
                constrained[False, String("Can't byte swap scalar type of size ", dtype.sizeof())]()


# NOTE: Mojo doesn't (yet) have default trait function implementations,
#       or extension functions, so to implement functions that work on all
#       structs that implement the BinaryReader trait, we (apparently)
#       need to define a new struct to hold the functions,
#       and then instantiate that struct on a reference when we want to call them.
struct BinaryDataReader[
    origin: Origin[mut=True],
    R: BinaryReader
]:
    var reader: Pointer[R, origin]

    fn __init__(out self, ref [origin] reader: R):
        self.reader = Pointer(to=reader)

    fn read_scalar[dtype: DType](self, out v: Scalar[dtype]) raises:
        constrained[
            dtype.sizeof() == 1,
            "For multi-byte scalars, use the function overload with an endian parameter"
        ]()
        v = 0
        _check_read[Scalar[dtype]](self.reader[].read_bytes(_as_mut_byte_span(v)))

    fn read_scalar[dtype: DType, endian: Endian](self, out v: Scalar[dtype]) raises:
        v = 0
        _check_read[Scalar[dtype]](self.reader[].read_bytes(_as_mut_byte_span(v)))
        _swap_bytes_if_needed[dtype, endian](v)

    fn read_u8(self, out v: UInt8) raises:
        v = self.read_scalar[DType.uint8]()

    fn read_i8(self, out v: Int8) raises:
        v = self.read_scalar[DType.int8]()

    fn read_u16[endian: Endian](self, out v: UInt16) raises:
        v = self.read_scalar[DType.uint16, endian]()

    fn read_i16[endian: Endian](self, out v: Int16) raises:
        v = self.read_scalar[DType.int16, endian]()

    fn read_u32[endian: Endian](self, out v: UInt32) raises:
        v = self.read_scalar[DType.uint32, endian]()

    fn read_i32[endian: Endian](self, out v: Int32) raises:
        v = self.read_scalar[DType.int32, endian]()

    fn read_u64[endian: Endian](self, out v: UInt64) raises:
        v = self.read_scalar[DType.uint64, endian]()

    fn read_i64[endian: Endian](self, out v: Int64) raises:
        v = self.read_scalar[DType.int64, endian]()

    fn read_f32[endian: Endian](self, out v: Float32) raises:
        v = self.read_scalar[DType.float32, endian]()

    fn read_f64[endian: Endian](self, out v: Float64) raises:
        v = self.read_scalar[DType.float64, endian]()

    # TODO: string?
    # TODO: boolean?
    # TODO: arrays?
