
trait BinaryWriter:
    fn write_bytes(mut self, bytes: Span[Byte]): ...
    fn write_scalar[dtype: DType](mut self, v: Scalar[dtype]): ...

    fn write_scalar[dtype: DType, endian: Endian](mut self, var v: Scalar[dtype]):
        swap_bytes_if_needed[dtype, endian](v)
        self.write_scalar(v)

    fn write_u8(mut self, var v: UInt8):
        self.write_scalar[DType.uint8](v)

    fn write_i8(mut self, var v: Int8):
        self.write_scalar[DType.int8](v)

    fn write_u16[endian: Endian](mut self, var v: UInt16):
        self.write_scalar[DType.uint16, endian](v)

    fn write_i16[endian: Endian](mut self, var v: Int16):
        self.write_scalar[DType.int16, endian](v)

    fn write_u32[endian: Endian](mut self, var v: UInt32):
        self.write_scalar[DType.uint32, endian](v)

    fn write_i32[endian: Endian](mut self, var v: Int32):
        self.write_scalar[DType.int32, endian](v)

    fn write_u64[endian: Endian](mut self, var v: UInt64):
        self.write_scalar[DType.uint64, endian](v)

    fn write_i64[endian: Endian](mut self, var v: Int64):
        self.write_scalar[DType.int64, endian](v)

    fn write_f32[endian: Endian](mut self, var v: Float32):
        self.write_scalar[DType.float32, endian](v)

    fn write_f64[endian: Endian](mut self, var v: Float64):
        self.write_scalar[DType.float64, endian](v)

    # TODO: string?
    # TODO: boolean?
    # TODO: arrays?
