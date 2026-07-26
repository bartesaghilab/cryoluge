
from cryoluge.math.units import Rad, Deg


struct Matrix[
    rows: Int,
    cols: Int,
    dtype: DType
](
    Copyable,
    Movable,
    Writable,
    Stringable,
    EqualityComparable
):
    var _values: InlineArray[Scalar[dtype], Self.num_elements]
    """Saved in row-major order."""

    comptime num_elements = rows*cols
    comptime D1 = Matrix[1,1,_]
    comptime D2 = Matrix[2,2,_]
    comptime D3 = Matrix[3,3,_]

    fn __init__(out self, *, uninitialized: Bool):
        self._values = InlineArray[Scalar[dtype], Self.num_elements](uninitialized=uninitialized)

    fn __init__(out self, *, fill: Scalar[dtype]):
        self._values = InlineArray[Scalar[dtype], Self.num_elements](fill=fill)

    fn __init__(out self, *, var row_major: InlineArray[Scalar[dtype],Self.num_elements]):
        self._values = row_major^

    @staticmethod
    fn row_major(out self: Self, *row_major: Scalar[dtype]):

        # check the size
        debug_assert(
            len(row_major) == Self.num_elements,
            "Expected ", Self.num_elements, " elements in ", rows, "x", cols, " matrix,"
            ", but got ", len(row_major), " elements instead."
        )

        self = Self(uninitialized=True)
        @parameter
        for i in range(Self.num_elements):
            self._values[i] = row_major[i]

    fn _index(self, row: Int, col: Int) -> Int:
        debug_assert(
            row >= 0 and row < rows and col >= 0 and col < cols,
            "Indices (", row, ",", col, ") out of range [0,", rows, ")x[0,", cols, ")"
        )
        return row*cols + col

    # accessors

    fn __getitem__(ref self, row: Int, col: Int) -> ref [self._values] Scalar[dtype]:
        return self._values[self._index(row, col)]

    fn __setitem__(mut self, row: Int, col: Int, v: Scalar[dtype]):
        self._values[self._index(row, col)] = v

    fn _vec[dim: Dimension](self, *, row: Int, out v: Vec[Scalar[dtype],dim]):
        v = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for c in range(cols):
            v[c] = self[row,c]

    fn vec(self: Matrix[rows,1,dtype], *, row: Int, out v: Vec.D1[Scalar[dtype]]):
        v = self._vec[Dimension.D1](row=row)

    fn vec(self: Matrix[rows,2,dtype], *, row: Int, out v: Vec.D2[Scalar[dtype]]):
        v = self._vec[Dimension.D2](row=row)

    fn vec(self: Matrix[rows,3,dtype], *, row: Int, out v: Vec.D3[Scalar[dtype]]):
        v = self._vec[Dimension.D3](row=row)

    fn _vec[dim: Dimension](self, *, col: Int, out v: Vec[Scalar[dtype],dim]):
        v = Vec[Scalar[dtype],dim](uninitialized=True)
        @parameter
        for r in range(rows):
            v[r] = self[r,col]

    fn vec(self: Matrix[1,cols,dtype], *, col: Int, out v: Vec.D1[Scalar[dtype]]):
        v = self._vec[Dimension.D1](col=col)

    fn vec(self: Matrix[2,cols,dtype], *, col: Int, out v: Vec.D2[Scalar[dtype]]):
        v = self._vec[Dimension.D2](col=col)

    fn vec(self: Matrix[3,cols,dtype], *, col: Int, out v: Vec.D3[Scalar[dtype]]):
        v = self._vec[Dimension.D3](col=col)

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

    fn __init__(out self: Self.D3[dtype], *, rotate_x: Rad[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_x(rotate_x)

    fn __init__(out self: Self.D3[dtype], *, rotate_x: Deg[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_x(rotate_x)

    fn set_rotate_x(mut self: Self.D3[dtype], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Self.D3[dtype].row_major(
            1, 0, 0,
            1, c, -s,
            1, s, c
        )

    fn set_rotate_x(mut self: Self.D3[dtype], angle: Deg[dtype]):
        self.set_rotate_x(angle.to_rad())

    fn __init__(out self: Self.D3[dtype], *, rotate_y: Rad[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_y(rotate_y)

    fn __init__(out self: Self.D3[dtype], *, rotate_y: Deg[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_y(rotate_y)

    fn set_rotate_y(mut self: Self.D3[dtype], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Self.D3[dtype].row_major(
            c, 0, s,
            0, 1, 0,
            -s, 0, c
        )

    fn set_rotate_y(mut self: Self.D3[dtype], angle: Deg[dtype]):
        self.set_rotate_y(angle.to_rad())

    fn __init__(out self: Self.D3[dtype], *, rotate_z: Rad[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_z(rotate_z)

    fn __init__(out self: Self.D3[dtype], *, rotate_z: Deg[dtype]):
        self = Self.D3[dtype](uninitialized=True)
        self.set_rotate_z(rotate_z)

    fn set_rotate_z(mut self: Self.D3[dtype], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Self.D3[dtype].row_major(
            c, -s, 0,
            s, c, 0,
            0, 0, 1
        )

    fn set_rotate_z(mut self: Self.D3[dtype], angle: Deg[dtype]):
        self.set_rotate_z(angle.to_rad())

    # modifiers

    fn transpose(mut self):
        @parameter
        for r in range(rows):
            @parameter
            for c in range(r):
                # LOL: this can't work on self things
                #swap(self[r,c], self[c,r])
                var s = self[r,c]
                self[r,c] = self[c,r]
                self[c,r] = s

    # operators

    fn __mul__[other_cols: Int](
        self,
        rhs: Matrix[cols,other_cols,dtype],
        out product: Matrix[rows,other_cols,dtype]
    ):
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

    fn __mul__[dim: Dimension, simd_width: Int](
        self,
        vec: Vec[SIMD[dtype,simd_width],dim],
        out result: Vec[SIMD[dtype,simd_width],dim]
    ):
        constrained[
            rows == dim.rank and cols == dim.rank,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim.rank,  ")")
        ]()

        result = Vec[SIMD[dtype,simd_width],dim](uninitialized=True)
        @parameter
        for d in range(dim.rank):
            var v = SIMD[dtype,simd_width](0)
            @parameter
            for i in range(dim.rank):
                v += self[d,i]*vec[i]
            result[d] = v

    fn __mul__[dim: Dimension, utype: UnitType](
        self,
        vec: Vec[Unit[utype,dtype],dim],
        out result: Vec[Unit[utype,dtype],dim]
    ):
        result = (self*vec.map_value()).map_unit[utype]()

    fn __eq__(self, other: Self) -> Bool:
        @parameter
        for i in range(Self.num_elements):
            if self._values[i] != other._values[i]:
                return False
        return True

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

    # display

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
