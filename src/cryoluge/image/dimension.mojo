
from std.sys.info import size_of
from std.memory import bitcast

from cryoluge.math import Vec
from cryoluge.math.dimension import unimplemented_dimension, expect_at_least_dim
from cryoluge.io import ByteBuffer, unsafe_span_bitcast


struct DimensionalBuffer[
    dim: Int,
    T: Copyable & Deinitable
](
    Copyable,
    Movable
):
    var _sizes: Self.Vec[.int]
    var _strides: Self.Vec[.int]
    """the element strides, not the byte strides"""
    var _buf: ByteBuffer

    comptime Vec = Vec[Self.dim,_]
    comptime _elem_size = size_of[Self.T]()

    def __init__(out self, sizes: Self.Vec[.int], *, alignment: Optional[Int] = None):
        """WARNING: produces un-initialized memory."""
        self._sizes = sizes.copy()
        comptime if Self.dim == 1:
            var sx = self._sizes.x()
            self._strides = Self.Vec[.int](x=1)
            self._buf = ByteBuffer(sx*Self._elem_size, alignment=alignment)
        elif Self.dim == 2:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            self._strides = Self.Vec[.int](x=1, y=sx)
            self._buf = ByteBuffer(sx*sy*Self._elem_size, alignment=alignment)
        elif Self.dim == 3:
            var sx = self._sizes.x()
            var sy = self._sizes.y()
            var sz = self._sizes.z()
            self._strides = Self.Vec[.int](x=1, y=sx, z=sx*sy)
            self._buf = ByteBuffer(sx*sy*sz*Self._elem_size, alignment=alignment)
        else:
            unimplemented_dimension[Self.dim]()

    @inline(.always)
    def dimension(self) -> Int:
        return Self.dim

    @inline(.always)
    def num_elements(self) -> Int:
        return self._sizes.product()

    @inline(.always)
    def sizes(self) -> ref [origin_of(self._sizes)] Self.Vec[.int]:
        return self._sizes

    @inline(.always)
    def span_bytes(
        ref self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Byte, origin_of(self._buf._alloc._alloc)]:
        return self._buf.span(start=start, length=length)
    
    @inline(.always)
    def span(
        ref self,
        *,
        start: Int = 0,
        length: Optional[Int] = None
    ) -> Span[Self.T, origin_of(self._buf._alloc._alloc)]:
        return unsafe_span_bitcast[Self.T](self.span_bytes(
            start=start*Self._elem_size,
            length=length.map(lambda (l: Int) -> Int: l*Self._elem_size)
        ))

    @inline(.always)
    def alignment(self) -> Int:
        return self._buf.alignment()

    @inline(.always)
    def offset(self, i: Self.Vec[.int], out offset: Int):
        
        comptime d_names = ["x", "y", "z"]

        offset = 0

        comptime for d in range(Self.dim):
            var coord = i[d]
            var size = self._sizes[d]
            debug_assert(
                coord >= 0 and coord < size,
                materialize[d_names[d]](), "=", coord, " out of range [0,", size, ")"
            )
            offset += coord*self._strides[d]

    @inline(.always)
    def maybe_offset(self, i: Self.Vec[.int]) -> Optional[Int]:

        var offset: Int = 0

        comptime for d in range(Self.dim):
            var coord = i[d]
            var size = self._sizes[d]
            if coord >= 0 and coord < size:
                offset += coord*self._strides[d]
            else:
                return None
        
        return offset

    @inline(.always)
    def __getitem__(self, *, i: Int, out v: Self.T):
        v = self.span(start=i, length=1)[0].copy()

    @inline(.always)
    def __getitem__(self, i: Self.Vec[.int], out v: Self.T):
        v = self[i=self.offset(i)]

    @inline(.always)
    def __getitem__(self, *, x: Int, out v: Self.T):
        expect_at_least_dim[Self.dim, 1]()
        v = self[Self.Vec(x=x)]

    @inline(.always)
    def __getitem__(self, *, x: Int, y: Int, out v: Self.T):
        expect_at_least_dim[Self.dim, 2]()
        v = self[Self.Vec(x=x, y=y)]

    @inline(.always)
    def __getitem__(self, *, x: Int, y: Int, z: Int, out v: Self.T):
        expect_at_least_dim[Self.dim, 3]()
        v = self[Self.Vec[.int](x=x, y=y, z=z)]

    @inline(.always)
    def __setitem__(mut self, *, i: Int, v: Self.T):
        self.span(start=i, length=1)[0] = v.copy()

    @inline(.always)
    def __setitem__(mut self, i: Self.Vec[.int], v: Self.T):
        self[i=self.offset(i)] = v

    @inline(.always)
    def __setitem__(mut self, *, x: Int, v: Self.T):
        expect_at_least_dim[Self.dim, 1]()
        self[Self.Vec[.int](x=x)] = v

    @inline(.always)
    def __setitem__(mut self, *, x: Int, y: Int, v: Self.T):
        expect_at_least_dim[Self.dim, 2]()
        self[Self.Vec[.int](x=x, y=y)] = v

    @inline(.always)
    def __setitem__(mut self, *, x: Int, y: Int, z: Int, v: Self.T):
        expect_at_least_dim[Self.dim, 3]()
        self[Self.Vec[.int](x=x, y=y, z=z)] = v

    @inline(.always)
    def get(self, i: Self.Vec[.int]) -> Optional[Self.T]:
        var offset = self.maybe_offset(i)
        if offset is None:
            return None
        return self[i=offset.value()]

    @inline(.always)
    def get[*, or_else: Self.T](self, i: Self.Vec[.int], out v: Self.T):

        var offset: Int = 0

        comptime for d in range(Self.dim):
            if i[d] >= 0 and i[d] < self._sizes[d]:
                offset += i[d]*self._strides[d]
            else:
                v = materialize[or_else]()
                return

        v = self[i=offset]

    def iterate(self, func: Some[def (i: Self.Vec[.int])]):
        self.sizes().iterate_over_sizes(func)
