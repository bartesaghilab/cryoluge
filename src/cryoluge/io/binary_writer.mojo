

trait BinaryWriter:
    fn write_bytes(mut self, bytes: Span[Byte]): ...


struct BinaryDataWriter[
    W: BinaryWriter,
    origin: Origin[mut=True]
]:
    var writer: Pointer[W, origin]

    fn __init__(out self, ref [origin] writer: W):
        self.writer = Pointer(to=writer)

    fn write_int[dtype: DType](mut self, v: Scalar[dtype]):
        constrained[
            dtype.sizeof() == 1,
            "For multi-byte integers, use the endian function overloads, or add an endian parameter"
        ]()
        self.writer[].write_bytes(v.as_bytes())

    fn write_int[dtype: DType, //, endian: Endian](mut self, v: Scalar[dtype]):
        self.writer[].write_bytes(v.as_bytes[big_endian = endian == Endian.Big]())

    fn write_int_be[dtype: DType, //](mut self, v: Scalar[dtype]):
        self.write_int[endian=Endian.Big](v)

    fn write_int_le[dtype: DType, //](mut self, v: Scalar[dtype]):
        self.write_int[endian=Endian.Little](v)
    
    fn write_float[dtype: DType, //](mut self, v: Scalar[dtype]):
        self.writer[].write_bytes(v.as_bytes())
