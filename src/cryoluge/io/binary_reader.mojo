
trait BinaryReader:
    fn read_bytes(mut self, buf: Span[mut=True, Byte]) raises -> Int: ...
    fn read_bytes_exact(mut self, but: Span[mut=True, Byte]) raises: ...
    fn read_scalar[dtype: DType](mut self, out v: Scalar[dtype]) raises: ...
    fn skip_bytes(mut self, size: Int) raises: ...
    fn skip_scalar[dtype: DType](mut self) raises: ...
    fn offset(self) -> UInt64: ...
    fn seek_to(mut self, offset: UInt64) raises: ...
    fn seek_by(mut self, offset: Int64) raises: ...

    fn read_scalar[dtype: DType, endian: Endian](mut self, out v: Scalar[dtype]) raises:
        v = self.read_scalar[dtype]()
        swap_bytes_if_needed[dtype, endian](v)

    fn read_u8(mut self, out v: UInt8) raises:
        v = self.read_scalar[DType.uint8]()

    fn read_i8(mut self, out v: Int8) raises:
        v = self.read_scalar[DType.int8]()

    fn read_u16[endian: Endian](mut self, out v: UInt16) raises:
        v = self.read_scalar[DType.uint16, endian]()

    fn read_i16[endian: Endian](mut self, out v: Int16) raises:
        v = self.read_scalar[DType.int16, endian]()

    fn read_u32[endian: Endian](mut self, out v: UInt32) raises:
        v = self.read_scalar[DType.uint32, endian]()

    fn read_i32[endian: Endian](mut self, out v: Int32) raises:
        v = self.read_scalar[DType.int32, endian]()

    fn read_u64[endian: Endian](mut self, out v: UInt64) raises:
        v = self.read_scalar[DType.uint64, endian]()

    fn read_i64[endian: Endian](mut self, out v: Int64) raises:
        v = self.read_scalar[DType.int64, endian]()

    fn read_f32[endian: Endian](mut self, out v: Float32) raises:
        v = self.read_scalar[DType.float32, endian]()

    fn read_f64[endian: Endian](mut self, out v: Float64) raises:
        v = self.read_scalar[DType.float64, endian]()

    # TODO: string?
    # TODO: boolean?
    # TODO: arrays?
