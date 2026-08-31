
from cryoluge.math import round
from cryoluge.math.units import Rad, Deg


struct Matrix[
    rows: Int,
    cols: Int,
    dtype: DType,
    simd_width: Int = 1
](
    Copyable,
    Movable,
    Writable,
    Stringable,
    EqualityComparable
):
    var _values: InlineArray[SIMD[dtype,simd_width], Self.num_elements]
    """Saved in row-major order."""

    comptime num_elements = rows*cols

    fn __init__(out self, *, uninitialized: Bool):
        self._values = InlineArray[SIMD[dtype,simd_width], Self.num_elements](uninitialized=uninitialized)

    fn __init__(out self, *, fill: SIMD[dtype,simd_width]):
        self._values = InlineArray[SIMD[dtype,simd_width], Self.num_elements](fill=fill)

    fn __init__(out self, *, var row_major: InlineArray[SIMD[dtype,simd_width],Self.num_elements]):
        self._values = row_major^

    @staticmethod
    fn row_major(out self: Self, *row_major: SIMD[dtype,simd_width]):

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

    @always_inline
    fn __getitem__(ref self, row: Int, col: Int) -> ref [self._values] SIMD[dtype,simd_width]:
        return self._values[self._index(row, col)]

    @always_inline
    fn __getitem__(ref self, *, slice: Int, out result: Matrix[rows,cols,dtype,1]):
        result = Matrix[rows,cols,dtype,1](uninitialized=True)
        @parameter
        for i in range(rows*cols):
            result._values[i] = self._values[i][slice]

    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, v: SIMD[dtype,simd_width]):
        self._values[self._index(row, col)] = v

    @always_inline
    fn __setitem__(mut self, *, slice: Int, v: Matrix[rows,cols,dtype,1]):
        @parameter
        for i in range(rows*cols):
            self._values[i][slice] = v._values[i]

    @always_inline
    fn _vec[dim: Int](self, *, row: Int, out v: Vec[dim,SIMD[dtype,simd_width]]):
        v = Vec[dim,SIMD[dtype,simd_width]](uninitialized=True)
        @parameter
        for c in range(cols):
            v[c] = self[row,c]

    @always_inline
    fn vec(self: Matrix[rows,1,dtype,simd_width], *, row: Int, out v: Vec[1,SIMD[dtype,simd_width]]):
        v = self._vec[1](row=row)

    @always_inline
    fn vec(self: Matrix[rows,2,dtype,simd_width], *, row: Int, out v: Vec[2,SIMD[dtype,simd_width]]):
        v = self._vec[2](row=row)

    @always_inline
    fn vec(self: Matrix[rows,3,dtype,simd_width], *, row: Int, out v: Vec[3,SIMD[dtype,simd_width]]):
        v = self._vec[3](row=row)

    @always_inline
    fn _vec[dim: Int](self, *, col: Int, out v: Vec[dim,SIMD[dtype,simd_width]]):
        v = Vec[dim,SIMD[dtype,simd_width]](uninitialized=True)
        @parameter
        for r in range(rows):
            v[r] = self[r,col]

    @always_inline
    fn vec(self: Matrix[1,cols,dtype,simd_width], *, col: Int, out v: Vec[1,SIMD[dtype,simd_width]]):
        v = self._vec[1](col=col)

    @always_inline
    fn vec(self: Matrix[2,cols,dtype,simd_width], *, col: Int, out v: Vec[2,SIMD[dtype,simd_width]]):
        v = self._vec[2](col=col)

    @always_inline
    fn vec(self: Matrix[3,cols,dtype,simd_width], *, col: Int, out v: Vec[3,SIMD[dtype,simd_width]]):
        v = self._vec[3](col=col)

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

    @always_inline
    fn __init__(out self: Matrix[2,2,dtype,simd_width], *, rotate: Rad[dtype]):
        self = Matrix[2,2,dtype,simd_width](uninitialized=True)
        self.set_rotate(rotate)

    @always_inline
    fn __init__(out self: Matrix[2,2,dtype,simd_width], *, rotate: Deg[dtype]):
        self = Matrix[2,2,dtype,simd_width](uninitialized=True)
        self.set_rotate(rotate)

    @always_inline
    fn set_rotate(mut self: Matrix[2,2,dtype,simd_width], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Matrix[2,2,dtype,simd_width].row_major(
            c, -s,
            s, c
        )

    @always_inline
    fn set_rotate(mut self: Matrix[2,2,dtype,simd_width], angle: Deg[dtype]):
        self.set_rotate(angle.to_rad())
    
    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_x: Rad[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_x(rotate_x)

    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_x: Deg[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_x(rotate_x)

    @always_inline
    fn set_rotate_x(mut self: Matrix[3,3,dtype,simd_width], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Matrix[3,3,dtype,simd_width].row_major(
            1, 0, 0,
            1, c, -s,
            1, s, c
        )

    @always_inline
    fn set_rotate_x(mut self: Matrix[3,3,dtype,simd_width], angle: Deg[dtype]):
        self.set_rotate_x(angle.to_rad())

    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_y: Rad[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_y(rotate_y)

    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_y: Deg[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_y(rotate_y)

    @always_inline
    fn set_rotate_y(mut self: Matrix[3,3,dtype,simd_width], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Matrix[3,3,dtype,simd_width].row_major(
            c, 0, s,
            0, 1, 0,
            -s, 0, c
        )

    @always_inline
    fn set_rotate_y(mut self: Matrix[3,3,dtype,simd_width], angle: Deg[dtype]):
        self.set_rotate_y(angle.to_rad())

    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_z: Rad[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_z(rotate_z)

    @always_inline
    fn __init__(out self: Matrix[3,3,dtype,simd_width], *, rotate_z: Deg[dtype]):
        self = Matrix[3,3,dtype,simd_width](uninitialized=True)
        self.set_rotate_z(rotate_z)

    @always_inline
    fn set_rotate_z(mut self: Matrix[3,3,dtype,simd_width], angle: Rad[dtype]):
        var s = angle.sin()
        var c = angle.cos()
        self = Matrix[3,3,dtype,simd_width].row_major(
            c, -s, 0,
            s, c, 0,
            0, 0, 1
        )

    @always_inline
    fn set_rotate_z(mut self: Matrix[3,3,dtype,simd_width], angle: Deg[dtype]):
        self.set_rotate_z(angle.to_rad())

    # modifiers

    @always_inline
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

    @always_inline
    fn transposed(self, out result: Self):
        result = Self(uninitialized=True)
        @parameter
        for r in range(rows):
            @parameter
            for c in range(cols):
                result[r,c] = self[c,r]

    # operators

    @always_inline
    fn __mul__[other_cols: Int](
        self,
        rhs: Matrix[cols,other_cols,dtype,simd_width],
        out product: Matrix[rows,other_cols,dtype,simd_width]
    ):
        product = Matrix[rows,other_cols,dtype,simd_width](uninitialized=True)
        @parameter
        for r in range(rows):
            @parameter
            for c in range(other_cols):
                var v = SIMD[dtype,simd_width](0)
                @parameter
                for i in range(cols):
                    v += self[r,i]*rhs[i,c]
                product[r,c] = v

    @always_inline
    fn __mul__[dim: Int, vec_simd_width: Int](
        self: Matrix[rows,cols,dtype,1],
        vec: Vec[dim,SIMD[dtype,vec_simd_width]],
        out result: Vec[dim,SIMD[dtype,vec_simd_width]]
    ):
        constrained[
            rows == dim and cols == dim,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim,  ")")
        ]()

        result = Vec[dim,SIMD[dtype,vec_simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var v = SIMD[dtype,vec_simd_width](0)
            @parameter
            for i in range(dim):
                v += self[d,i]*vec[i]
            result[d] = v

    @always_inline
    fn __mul__[dim: Int](
        self: Self,
        vec: Vec[dim,SIMD[dtype,simd_width]],
        out result: Vec[dim,SIMD[dtype,simd_width]]
    ):
        constrained[
            rows == dim and cols == dim,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim,  ")")
        ]()

        result = Vec[dim,SIMD[dtype,simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var v = SIMD[dtype,simd_width](0)
            @parameter
            for i in range(dim):
                v += self[d,i]*vec[i]
            result[d] = v

    @always_inline
    fn __mul__(
        self,
        f: Scalar[dtype],
        out result: Self
    ):
        result = Self(uninitialized=True)
        @parameter
        for r in range(rows):
            @parameter
            for c in range(cols):
                result[r,c] = self[r,c]*f
    
    @always_inline
    fn __mul__[dim: Int, utype: UnitType](
        self: Matrix[rows,cols,dtype,1],
        vec: Vec[dim,Unit[utype,dtype]],
        out result: Vec[dim,Unit[utype,dtype]]
    ):
        result = (self*vec.map_value()).map_unit[utype]()

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        @parameter
        for i in range(Self.num_elements):
            if self._values[i] != other._values[i]:
                return False
        return True

    # other math

    @always_inline
    fn mul_transpose[dim: Int, vec_simd_width: Int](
        self: Matrix[rows,cols,dtype,1],
        vec: Vec[dim,SIMD[dtype,vec_simd_width]],
        out result: Vec[dim,SIMD[dtype,vec_simd_width]]
    ):
        constrained[
            rows == dim and cols == dim,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim,  ")")
        ]()

        result = Vec[dim,SIMD[dtype,vec_simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var v = SIMD[dtype,vec_simd_width](0)
            @parameter
            for i in range(dim):
                v += self[i,d]*vec[i]
            result[d] = v

    @always_inline
    fn mul_transpose[dim: Int](
        self: Self,
        vec: Vec[dim,SIMD[dtype,simd_width]],
        out result: Vec[dim,SIMD[dtype,simd_width]]
    ):
        constrained[
            rows == dim and cols == dim,
            String("Matrix size (", rows, ", ", cols, ") doesn't match vector size (", dim,  ")")
        ]()

        result = Vec[dim,SIMD[dtype,simd_width]](uninitialized=True)
        @parameter
        for d in range(dim):
            var v = SIMD[dtype,simd_width](0)
            @parameter
            for i in range(dim):
                v += self[i,d]*vec[i]
            result[d] = v

    @always_inline
    fn __round__(self, digits: Int, out result: Self):
        result = Matrix[rows,cols,dtype,simd_width](uninitialized=True)
        @parameter
        for i in range(Self.num_elements):
            result._values[i] = self._values[i].__round__(digits)

    @always_inline
    fn round[digits: Int](self, out result: Self):
        result = Matrix[rows,cols,dtype,simd_width](uninitialized=True)
        @parameter
        for i in range(Self.num_elements):
            result._values[i] = round[digits](self._values[i])

    # conversion

    @always_inline
    fn map[
        out_dtype: DType,
        //,
        mapper: fn(SIMD[dtype,simd_width]) capturing -> SIMD[out_dtype,simd_width]
    ](self, out mat: Matrix[rows,cols,out_dtype,simd_width]):
        mat = Matrix[rows,cols,out_dtype,simd_width](uninitialized=True)
        @parameter
        for i in range(Self.num_elements):
            mat._values[i] = mapper(self._values[i])

    @always_inline
    fn map_scalar[out_dtype: DType](self, out result: Matrix[rows,cols,out_dtype,simd_width]):
        @parameter
        fn func(v: SIMD[dtype,simd_width], out mapped: SIMD[out_dtype,simd_width]):
            mapped = SIMD[out_dtype,simd_width](v)
        result = self.map[mapper=func]()
    
    @always_inline
    fn map_float32(self: Matrix[rows,cols,DType.float32,simd_width], out result: Matrix[rows,cols,DType.float32,simd_width]):
        result = self.map_scalar[DType.float32]()

    @always_inline
    fn map_float64(self: Matrix[rows,cols,DType.float64,simd_width], out result: Matrix[rows,cols,DType.float64,simd_width]):
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
