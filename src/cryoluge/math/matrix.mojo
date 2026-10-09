
from cryoluge.math import round
from cryoluge.math.vec import Vec  # TODO: why can't we import this from `cryoluge.math`?
from cryoluge.math.units import Rad, Deg


struct Matrix[
    rows: Int,
    cols: Int,
    dtype: DType,
    slen: SIMDLength = 1
](
    Copyable,
    Movable,
    Writable,
    Equatable
# TODO: Mojo's constaint system isn't quite fully-baked yet
#) where rows > 0 and cols > 0:
):

    var _values: Self.Arr
    """Saved in row-major order."""

    comptime num_elements = Self.rows*Self.cols
    comptime Arr = Array[Self.V, Self.num_elements]
    comptime V = SIMD[Self.dtype, Self.slen]
    comptime WithRows = Matrix[_, Self.cols, Self.dtype, Self.slen]
    comptime WithCols = Matrix[Self.rows, _, Self.dtype, Self.slen]
    comptime WithRowsCols = Matrix[_, _, Self.dtype, Self.slen]
    comptime WithDType = Matrix[Self.rows, Self.cols, _, Self.slen]
    comptime WithSlen = Matrix[Self.rows, Self.cols, Self.dtype, _]

    @inline(.always)
    def __init__(out self, *, uninitialized: Bool):
        self._values = Self.Arr(uninitialized=uninitialized)

    @inline(.always)
    def __init__(out self, *, fill: Self.V):
        self._values = Self.Arr(fill=fill)

    @inline(.always)
    def __init__(out self, var row_major: Self.Arr):
        """Constructs a matrix from an Array in row-major order."""
        self._values = row_major^

    @implicit
    @inline(.always)
    def __init__[
        *,
        __literal_size__: Int
    ](
        out self,
        var *elems: Self.V,
        __list_literal__: NoneType = None,
    ):
        """Constructs a matrix from a list literal in row-major order."""
        comptime assert __literal_size__ == Self.num_elements,
            t"""
            Expected Array with {Self.num_elements} element(s) for {Self.rows}x{Self.cols} Matrix,
            but got {__literal_size__} instead.
            """

        self = Self(uninitialized=True)
        comptime for i in range(__literal_size__):
            self._values[i] = elems[i]

    @staticmethod
    def _index[row: Int, col: Int]() -> Int:
        comptime assert row >= 0 and row < Self.rows and col >= 0 and col < Self.cols,
            t"Indices ({row},{col}) out of range [0,{Self.rows})x[0,{Self.cols})"
        return row*Self.cols + col

    def _index(self, row: Int, col: Int) -> Int:
        debug_assert(
            row >= 0 and row < Self.rows and col >= 0 and col < Self.cols,
            t"Indices ({row},{col}) out of range [0,{Self.rows})x[0,{Self.cols})"
        )
        return row*Self.cols + col

    # accessors

    @inline(.always)
    def __getitem_param__[row: Int, col: Int](ref self) -> ref [self._values] Self.V:
        return self._values[Self._index[row, col]()]

    @inline(.always)
    def __getitem__(ref self, row: Int, col: Int) -> ref [self._values] Self.V:
        return self._values[self._index(row, col)]

    @inline(.always)
    def __getitem_param__[*, slice: Int](self, out result: Self.WithSlen[1]):
        result = Self.WithSlen[1](uninitialized=True)
        comptime for i in range(Self.num_elements):
            result._values[i] = self._values[i][slice]

    @inline(.always)
    def __getitem__(ref self, *, slice: Int, out result: Self.WithSlen[1]):
        result = Self.WithSlen[1](uninitialized=True)
        comptime for i in range(Self.num_elements):
            result._values[i] = self._values[i][slice]

    @inline(.always)
    def __setitem_param__[*, slice: Int](mut self, v: Self.WithSlen[1]):
        comptime for i in range(Self.num_elements):
            self._values[i][slice] = v._values[i]

    @inline(.always)
    def __setitem__(mut self, *, slice: Int, v: Self.WithSlen[1]):
        comptime for i in range(Self.num_elements):
            self._values[i][slice] = v._values[i]

    @inline(.always)
    def vec(self, *, row: Int, out v: Vec[Self.cols,Self.dtype,Self.slen]):
        v = Vec[Self.cols,Self.dtype,Self.slen](uninitialized=True)
        comptime for c in range(Self.cols):
            v[c] = self[row,c]

    @inline(.always)
    def vec(self, *, col: Int, out v: Vec[Self.rows,Self.dtype,Self.slen]):
        v = Vec[Self.rows,Self.dtype,Self.slen](uninitialized=True)
        comptime for r in range(Self.rows):
            v[r] = self[r,col]

    # setters

    @staticmethod
    def identity(out self: Self):
        self = Self(uninitialized=True)
        self.set_identity()

    def set_identity(mut self):
        comptime for r in range(Self.rows):
            comptime for c in range(Self.cols):
                comptime if r == c:
                    self[r,c] = 1
                else:
                    self[r,c] = 0

    @inline(.always)
    def __init__(out self, *, rotate: Rad[Self.V])
        where Self.rows == 2 and Self.cols == 2 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate(rotate)

    @inline(.always)
    def __init__(out self, *, rotate: Deg[Self.V])
        where Self.rows == 2 and Self.cols == 2 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate(rotate)

    @inline(.always)
    def set_rotate(mut self, angle: Rad[Self.V])
        where Self.dtype.is_floating_point():

        var s = angle.sin()
        var c = angle.cos()
        self = [
            c, -s,
            s, c
        ]

    @inline(.always)
    def set_rotate(mut self, angle: Deg[Self.V])
        where Self.dtype.is_floating_point():

        self.set_rotate(angle.to_rad())
    
    @inline(.always)
    def __init__(out self, *, rotate_x: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_x(rotate_x)

    @inline(.always)
    def __init__(out self, *, rotate_x: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_x(rotate_x)

    @inline(.always)
    def set_rotate_x(mut self, angle: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        var s = angle.sin()
        var c = angle.cos()
        self = [
            1, 0, 0,
            1, c, -s,
            1, s, c
        ]

    @inline(.always)
    def set_rotate_x(mut self, angle: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self.set_rotate_x(angle.to_rad())

    @inline(.always)
    def __init__(out self, *, rotate_y: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_y(rotate_y)

    @inline(.always)
    def __init__(out self, *, rotate_y: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_y(rotate_y)

    @inline(.always)
    def set_rotate_y(mut self, angle: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        var s = angle.sin()
        var c = angle.cos()
        self = [
            c, 0, s,
            0, 1, 0,
            -s, 0, c
        ]

    @inline(.always)
    def set_rotate_y(mut self, angle: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self.set_rotate_y(angle.to_rad())

    @inline(.always)
    def __init__(out self, *, rotate_z: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_z(rotate_z)

    @inline(.always)
    def __init__(out self, *, rotate_z: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self = Self(uninitialized=True)
        self.set_rotate_z(rotate_z)

    @inline(.always)
    def set_rotate_z(mut self, angle: Rad[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        var s = angle.sin()
        var c = angle.cos()
        self = [
            c, -s, 0,
            s, c, 0,
            0, 0, 1
        ]

    @inline(.always)
    def set_rotate_z(mut self, angle: Deg[Self.V])
        where Self.rows == 3 and Self.cols == 3 and Self.dtype.is_floating_point():

        self.set_rotate_z(angle.to_rad())

    # modifiers

    @inline(.always)
    def transpose(mut self):

        # just in case ...
        comptime assert Self.rows == Self.cols,
            t"Can only transpose a square matrix in place, not {Self.rows}x{Self.cols}"

        comptime for r in range(Self.rows):
            comptime for c in range(r):
                # LOL: this can't work on self things
                #swap(self[r,c], self[c,r])
                var s = self[r,c]
                self[r,c] = self[c,r]
                self[c,r] = s

    @inline(.always)
    def transposed(self, out result: Self.WithRowsCols[Self.cols,Self.rows]):
        result = Self.WithRowsCols[Self.cols,Self.rows](uninitialized=True)
        comptime for r in range(Self.rows):
            comptime for c in range(Self.cols):
                result[c,r] = self[r,c]

    # operators

    @inline(.always)
    def __mul__[other_cols: Int](
        self,
        rhs: Self.WithRowsCols[Self.cols,other_cols],
        out product: Self.WithRowsCols[Self.rows,other_cols]
    ) where other_cols > 0:
        """Multiply two matrices."""

        product = Self.WithRowsCols[Self.rows,other_cols](uninitialized=True)
        comptime for r in range(Self.rows):
            comptime for c in range(other_cols):
                var v = Self.V(0)
                comptime for i in range(Self.cols):
                    v += self[r,i]*rhs[i,c]
                product[r,c] = v

    @inline(.always)
    def __mul__[dim: Int, vec_slen: SIMDLength](
        self,
        vec: Vec[dim,Self.dtype,vec_slen],
        out result: Vec[dim,Self.dtype,vec_slen]
    ) where dim > 0 and Self.rows == Self.cols and Self.slen == 1:
        """Multiply a square matrix and a vector (or vectors)."""

        comptime assert Self.rows == dim and Self.cols == dim,
            t"Matrix size ({Self.rows}x{Self.cols}) doesn't match vector size ({dim})"

        result = Vec[dim, Self.dtype,vec_slen](uninitialized=True)
        comptime for d in range(dim):
            var v = SIMD[Self.dtype,vec_slen](0)
            comptime for i in range(dim):
                var s = rebind[Scalar[Self.dtype]]( self[d,i] )
                v += s*vec[i]
            result[d] = v

    @inline(.always)
    def __mul__[dim: Int](
        self,
        vec: Vec[dim,Self.dtype,Self.slen],
        out result: Vec[dim,Self.dtype,Self.slen]
    ) where dim > 0 and Self.rows == Self.cols:
        """Multiply a square matrix (or matrices) and a vector (or vectors)."""

        comptime assert Self.rows == dim and Self.cols == dim,
            t"Matrix size ({Self.rows}x{Self.cols}) doesn't match vector size ({dim})"

        result = Vec[dim,Self.dtype,Self.slen](uninitialized=True)
        comptime for d in range(dim):
            var v = Self.V(0)
            comptime for i in range(dim):
                v += self[d,i]*vec[i]
            result[d] = v

    @inline(.always)
    def __mul__(
        self,
        f: Self.V,
        out result: Self
    ):
        result = Self(uninitialized=True)
        comptime for r in range(Self.rows):
            comptime for c in range(Self.cols):
                result[r,c] = self[r,c]*f
    
    @inline(.always)
    def __eq__(self, other: Self) -> Bool:
        comptime for i in range(Self.num_elements):
            if self._values[i] != other._values[i]:
                return False
        return True

    # other math

    @inline(.always)
    def mul_transpose[dim: Int, vec_slen: SIMDLength](
        self,
        vec: Vec[dim,Self.dtype,vec_slen],
        out result: Vec[dim,Self.dtype,vec_slen]
    ) where dim > 0 and Self.rows == Self.cols and Self.slen == 1:
        """Multiply the transpose of a square matrix and a vector (or vectors)."""

        comptime assert Self.rows == dim and Self.cols == dim,
            t"Matrix size ({Self.rows}x{Self.cols}) doesn't match vector size ({dim})"

        result = Vec[dim,Self.dtype,vec_slen](uninitialized=True)
        comptime for d in range(dim):
            var v = SIMD[Self.dtype,vec_slen](0)
            comptime for i in range(dim):
                var s = rebind[Scalar[Self.dtype]]( self[i,d] )
                v += s*vec[i]
            result[d] = v

    @inline(.always)
    def mul_transpose[dim: Int](
        self: Self,
        vec: Vec[dim,Self.dtype,Self.slen],
        out result: Vec[dim,Self.dtype,Self.slen]
    ) where dim > 0 and Self.rows == Self.cols:
        """Multiply the transpose of a square matrix (or matrices) and a vector (or vectors)."""

        comptime assert Self.rows == dim and Self.cols == dim,
            t"Matrix size ({Self.rows}x{Self.cols}) doesn't match vector size ({dim})"

        result = Vec[dim,Self.dtype,Self.slen](uninitialized=True)
        comptime for d in range(dim):
            var v = Self.V(0)
            comptime for i in range(dim):
                v += self[i,d]*vec[i]
            result[d] = v

    @inline(.always)
    def __round__(self, digits: Int, out result: Self):
        result = Self(uninitialized=True)
        comptime for i in range(Self.num_elements):
            result._values[i] = self._values[i].__round__(digits)

    @inline(.always)
    def round[digits: Int](self, out result: Self):
        result = Self(uninitialized=True)
        comptime for i in range(Self.num_elements):
            result._values[i] = round[digits](self._values[i])

    # conversion

    @inline(.always)
    def map[
        _dtype: DType,
        _slen: SIMDLength,
        //
    ](
        self,
        f: Some[def(Self.V) -> SIMD[_dtype,_slen]],
        out mat: Matrix[Self.rows,Self.cols,_dtype,_slen]
    ):
        mat = Matrix[Self.rows,Self.cols,_dtype,_slen](uninitialized=True)
        comptime for i in range(Self.num_elements):
            mat._values[i] = f(self._values[i])

    @inline(.always)
    def map_dtype[_dtype: DType](self, out result: Self.WithDType[_dtype]):
        result = self.map(lambda (v: Self.V) -> SIMD[_dtype,Self.slen]: SIMD[_dtype,Self.slen](v))

    # display

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Matrix[{Self.rows}x{Self.cols}]:")
        comptime for r in range(Self.rows):
            writer.write("\n  ")
            comptime for c in range(Self.cols):
                writer.write("  ", self[r,c])
