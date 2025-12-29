
from math import sin, cos, pi


struct Matrix[
    rows: Int,
    cols: Int,
    dtype: DType
](
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var _values: InlineArray[Scalar[dtype], Self.num_elements]

    comptime num_elements = rows*cols
    comptime D3 = Matrix[3,3,_]

    fn __init__(out self, *, uninitialized: Bool):
        self._values = InlineArray[Scalar[dtype], Self.num_elements](uninitialized=uninitialized)

    fn __init__(out self, *, fill: Scalar[dtype]):
        self._values = InlineArray[Scalar[dtype], Self.num_elements](fill=fill)

    fn _index(self, row: Int, col: Int) -> Int:
        debug_assert(
            row >= 0 and row < rows and col >= 0 and col < cols,
            "Indices (", row, ",", col, ") out of range [0,", rows, ")x[0,", cols, ")"
        )
        return row*cols + col

    fn __getitem__(ref self, row: Int, col: Int) -> ref [self._values] Scalar[dtype]:
        return self._values[self._index(row, col)]

    fn __getitem__(self, row: Int, out v: InlineArray[Scalar[dtype],cols]):
        v = InlineArray[Scalar[dtype],cols](uninitialized=True)
        @parameter
        for c in range(cols):
            v[c] = self[row, c]

    fn __setitem__(mut self, row: Int, col: Int, v: Scalar[dtype]):
        self._values[self._index(row, col)] = v

    fn __setitem__(mut self, row: Int, v: InlineArray[Scalar[dtype],cols]):
        @parameter
        for c in range(cols):
            self[row, c] = v[c]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Matrix[", rows, ", ", cols, "]:")
        @parameter
        for r in range(rows):
            writer.write("\n  ")
            @parameter
            for c in range(cols):
                writer.write("  ", self[r,c])

    fn __str__(self) -> String:
        return String.write(self)

    # setters

    @staticmethod
    fn identity(out self: Self):
        self = Self(uninitialized=True)
        self.set_identity()

    fn set_identity(mut self):
        @parameter
        for r in range(rows):
            @parameter
            for c in range(cols):
                @parameter
                if r == c:
                    self[r,c] = 1
                else:
                    self[r,c] = 0

    fn __init__(out self: Self.D3[dtype], *, rotate_x_rad: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_x(rad=rotate_x_rad)

    fn __init__(out self: Self.D3[dtype], *, rotate_x_deg: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_x(deg=rotate_x_deg)

    fn set_rotate_x(mut self: Self.D3[dtype], *, rad: Scalar[dtype]):
        var s = sin(rad)
        var c = cos(rad)
        self[0] = InlineArray[Scalar[dtype],3](1, 0, 0)
        self[1] = InlineArray[Scalar[dtype],3](1, c, -s)
        self[2] = InlineArray[Scalar[dtype],3](1, s, c)

    fn set_rotate_x(mut self: Self.D3[dtype], *, deg: Scalar[dtype]):
        self.set_rotate_x(rad=deg_to_rad(deg=deg))

    fn __init__(out self: Self.D3[dtype], *, rotate_y_rad: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_y(rad=rotate_y_rad)

    fn __init__(out self: Self.D3[dtype], *, rotate_y_deg: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_y(deg=rotate_y_deg)

    fn set_rotate_y(mut self: Self.D3[dtype], *, rad: Scalar[dtype]):
        var s = sin(rad)
        var c = cos(rad)
        self[0] = InlineArray[Scalar[dtype],3](c, 0, s)
        self[1] = InlineArray[Scalar[dtype],3](0, 1, 0)
        self[2] = InlineArray[Scalar[dtype],3](-s, 0, c)

    fn set_rotate_y(mut self: Self.D3[dtype], *, deg: Scalar[dtype]):
        self.set_rotate_y(rad=deg_to_rad(deg=deg))

    fn __init__(out self: Self.D3[dtype], *, rotate_z_rad: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_z(rad=rotate_z_rad)

    fn __init__(out self: Self.D3[dtype], *, rotate_z_deg: Scalar[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_z(deg=rotate_z_deg)

    fn set_rotate_z(mut self: Self.D3[dtype], *, rad: Scalar[dtype]):
        var s = sin(rad)
        var c = cos(rad)
        self[0] = InlineArray[Scalar[dtype],3](c, -s, 0)
        self[1] = InlineArray[Scalar[dtype],3](s, c, 0)
        self[2] = InlineArray[Scalar[dtype],3](0, 0, 1)

    fn set_rotate_z(mut self: Self.D3[dtype], *, deg: Scalar[dtype]):
        self.set_rotate_z(rad=deg_to_rad(deg=deg))

    # math

    fn __mul__[other_cols: Int](self, rhs: Matrix[cols,other_cols,dtype], out product: Matrix[rows,other_cols,dtype]):
        product = Matrix[rows,other_cols,dtype](uninitialized=True)
        @parameter
        for r in range(rows):
            @parameter
            for c in range(other_cols):
                var v = Scalar[dtype](0)
                @parameter
                for i in range(cols):
                    v += self[r,i]*rhs[i,c]
                product[r,c] = v

    fn __mul__[dim: Dimension](
        self: Matrix[rows,cols,DType.float32],
        vec: Vec[Float32,dim],
        out result: Vec[Float32,dim]
    ):
        constrained[
            rows == dim.rank and cols == dim.rank,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim.rank,  ")")
        ]()

        result = Vec[Float32,dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            var v = Float32(0)
            @parameter
            for i in range(dim.rank):
                v += self[d,i]*vec[i]
            result[d] = v

    # conversion

    fn map[
        out_dtype: DType,
        //,
        mapper: fn(Scalar[dtype]) capturing -> Scalar[out_dtype]
    ](self, out mat: Matrix[rows,cols,out_dtype]):
        mat = Matrix[rows,cols,out_dtype](uninitialized=True)
        @parameter
        for i in range(Self.num_elements):
            mat._values[i] = mapper(self._values[i])

    fn map_scalar[out_dtype: DType](self, out result: Matrix[rows,cols,out_dtype]):
        @parameter
        fn func(v: Scalar[dtype], out mapped: Scalar[out_dtype]):
            mapped = Scalar[out_dtype](v)
        result = self.map[mapper=func]()
    
    fn map_float32(self: Matrix[rows,cols,DType.float32], out result: Matrix[rows,cols,DType.float32]):
        result = self.map_scalar[DType.float32]()

    fn map_float64(self: Matrix[rows,cols,DType.float64], out result: Matrix[rows,cols,DType.float64]):
        result = self.map_scalar[DType.float64]()
