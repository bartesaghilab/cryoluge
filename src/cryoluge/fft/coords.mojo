
from cryoluge.math import Vec, is_odd


struct FFTCoords[dim: Int](
    Copyable,
    Movable
):
    """
    Keeps track of coordinate spaces related to FFTs.

    Follows the "standard" order:

    For 1D transforms, where A contains the FFT of n real values:
        A contains n/2+1 elements
        A[0] contains the zero-frequency term
        A[1:n/2] contains the positive-frequency terms, in order of ascending frequency

    In higher dimensions, where A[d] contains the d>1 dimension of n real values:
        A[d] contains n elements
        A[d][0] contains the zero-frequency term
        if n is even:
            A[d][1:n/2) contains the positive-frequency terms, in order of ascending frequency
            A[d][n/2:n) contains the negative-frequency terms, in order of decreasing absolute frequency
        if n is odd:
            A[d][1:n/2+1) contains the positive-frequency terms, in order of ascending frequency
            A[d][n/2+1:n) contains the negative-frequency terms, in order of decreasing absolute frequency
    """

    var _sizes_real: Self.Vec[.int]

    comptime Vec = Vec[Self.dim, _]

    def __init__(out self, sizes_real: Self.Vec[.int]):
        self._sizes_real = sizes_real.copy()

    def full(self) -> FFTCoordsFull[Self.dim]:
        return FFTCoordsFull(self._sizes_real)

    @inline(.always)
    def sizes_real(self) -> ref [origin_of(self._sizes_real)] Self.Vec[.int]:
        return self._sizes_real

    @inline(.always)
    def size_fourier[d: Int](self, out size_fourier: Int):
        comptime if d == 0:
            size_fourier = (self.sizes_real()[d] >> 1) + 1
        else:
            size_fourier = self.sizes_real()[d]

    def sizes_fourier(self, out sizes_fourier: Self.Vec[.int]):
        sizes_fourier = Self.Vec[.int](uninitialized=True)
        comptime for d in range(Self.dim):
            sizes_fourier[d] = self.size_fourier[d]()

    @inline(.always)
    def _pivot[d: Int](self, out pivot: Int):
        """The first image index that maps to a negative frequency."""
        comptime if d == 0:
            comptime assert False, "No pivot for x"
        else:
            pivot = (self.sizes_real()[d] + 1) >> 1
    
    @inline(.always)
    def fmin[d: Int](self, out fmin: Int):
        comptime if d == 0:
            fmin = 1 - self.size_fourier[d]()
        else:
            fmin = self._pivot[d]() - self.sizes_real()[d]

    @inline(.always)
    def fmin(self, out fmin: Self.Vec[.int]):
        """Returns the lower bound (inclusive) on fourier coordinates."""
        fmin = Self.Vec[.int](uninitialized=True)
        comptime for d in range(Self.dim):
            fmin[d] = self.fmin[d]()

    @inline(.always)
    def fmin_pos[d: Int](self, out fmin: Int):
        comptime if d == 0:
            fmin = 0
        else:
            fmin = self.fmin[d]()

    @inline(.always)
    def fmin_pos(self, out fmin: Self.Vec[.int]):
        """Returns the lower bound (inclusive) on fourier coordinates for the positive x-halfspace."""
        fmin = Self.Vec[.int](uninitialized=True)
        comptime for d in range(Self.dim):
            fmin[d] = self.fmin_pos[d]()

    @inline(.always)
    def fmax[d: Int](self, out fmax: Int):
        comptime if d == 0:
            fmax = self.size_fourier[d]() - 1
        else:
            fmax = self._pivot[d]() - 1

    @inline(.always)
    def fmax(self, out fmax: Self.Vec[.int]):
        """Returns the upper bound (inclusive) on fourier coordinates."""
        fmax = Self.Vec[.int](uninitialized=True)
        comptime for d in range(Self.dim):
            fmax[d] = self.fmax[d]()

    @inline(.always)
    def needs_conjugation(self, *, f: Self.Vec[.int]) -> Bool:
        return f.x() < 0

    @inline(.always)
    def f_in_range[d: Int](self, f: Int, out in_range: Bool):
        in_range = f >= self.fmin[d]() and f <= self.fmax[d]()

    @inline(.always)
    def f_in_range(self, f: Self.Vec[.int]) -> Bool:
        comptime for d in range(Self.dim):
            if not self.f_in_range[d](f[d]):
                return False
        return True

    @inline(.always)
    def f2i(self, f: Self.Vec[.int], out i: Self.Vec[.int]):
        """
        Converts fourier coordinates to image coordinates.
        Maps negative x fourier coordinates to the positive side.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.

        Precondition: f is in range
        """

        i = Self.Vec[.int](uninitialized=True)

        var _f = f.copy()
        if self.needs_conjugation(f=f):
            _f = -_f

        i[0] = _f[0]
        comptime for d in range(1, Self.dim):
            if _f[d] < 0:
                i[d] = _f[d] + self.size_fourier[d]()
            else:
                i[d] = _f[d]

    @inline(.always)
    def maybe_f2i(self, f: Self.Vec[.int], out i: Optional[Self.Vec[.int]]):
        """
        If the Fourier coordinates are in-range, this function converts them to image coordinates.
        Otherwise, returns None.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.
        """

        if self.f_in_range(f):
            i = self.f2i(f)
        else:
            i = None

    @inline(.always)
    def f2i_contiguous(self, f: Self.Vec[.int], out i: Self.Vec[.int]):
        
        i = f.copy()

        # use contiguous addressing, instead of the bifurcated thing in f2i
        if self.needs_conjugation(f=f):
            i = -i
        i -= self.fmin_pos()
        i %= self.sizes_fourier()

    @inline(.always)
    def maybe_f2i_contiguous(self, f: Self.Vec[.int], out i: Optional[Self.Vec[.int]]):
        if self.f_in_range(f):
            i = self.f2i_contiguous(f)
        else:
            return None
    
    @inline(.always)
    def i2f(self, i: Self.Vec[.int], out f: Self.Vec[.int]):
        """
        Converts image coordinates to fourier coordinates.
        
        Precondition: i is in range
        """

        f = i.copy()

        # no transformation needed for x
        # but apply the transformation to y,z
        comptime for d in range(1, Self.dim):
            if i[d] >= self._pivot[d]():
                f[d] -= self.size_fourier[d]()

    @inline(.always)
    def i2f_contiguous(self, i: Self.Vec[.int], out f: Self.Vec[.int]):
        # use contiguous addressing, instead of the bifurcated thing in i2f
        f = i + self.fmin_pos()

    @inline(.always)
    def sizes_voxel_norm[dtype: DType](self, out sizes: self.Vec[dtype]):
        """
        Returns the sizes of one pixel/voxel in normalized frequency coordinates.
        WARNING: This operation can be (relatively) slow.
                 Caching the result can give noticeable speedups.
                 See FrequencyLimitsChecker.
        """
        sizes = 1/self.sizes_real().map_dtype[dtype]()

    @inline(.always)
    def f_norm[dtype: DType](
        self,
        *,
        f: Self.Vec[.int],
        out f_norm: Self.Vec[dtype]
    ):
        """Returns the normalized frequencies of the Fourier coordinates."""
        var f_dt = f.map_dtype[dtype]()
        f_norm = self.f_norm(f=f_dt)

    @inline(.always)
    def f_norm[dtype: DType](
        self,
        *,
        f: Self.Vec[dtype],
        out f_norm: Self.Vec[dtype]
    ):
        """Returns the normalized frequencies of the Fourier coordinates."""
        # NOTE: multipliying by the reciprical is measurably faster than just doing division
        #       and also matches the roundoff error of csp1
        f_norm = f*self.sizes_voxel_norm[dtype]()

    def f_norm[dtype: DType](
        self,
        *,
        i: Self.Vec[.int],
        out f_norm: Self.Vec[dtype]
    ):
        """Returns the normalized frequencies of the image coordinates."""
        f_norm = self.f_norm[dtype](f=self.i2f(i))

    def freq_edge[dtype: DType](self, out freq: Scalar[dtype]):
        """
        Returns the positive frequency at the edge of the fourier area,
        which is always 0.5, since the frequencies are normalized by the sizes.
        """
        freq = Scalar[dtype](0.5)


struct FFTCoordsFull[dim: Int](
    Copyable,
    Movable
):
    """
    A tool to handle FFT coordinate spaces that store both negative and positive x freqencies.

    For 1D transforms, where A contains the FFT of n real values:
        A contains n/2*2+1 elements
        A[0] contains the zero-frequency term
        A[1:n/2] contains the positive-frequency terms, in order of ascending frequency
        A[n/2+1:n/2*2+1) contains the negative-frequency terms, in order of ascending frequency
    
    In higher dimensions, the coordinates are handled the same as in FFTCoords
    """

    var _sizes_real: Self.Vec[.int]

    comptime Vec = Vec[Self.dim,_]

    def __init__(out self, sizes_real: Self.Vec[.int]):
        self._sizes_real = sizes_real.copy()

    def half(self) -> FFTCoords[Self.dim]:
        return FFTCoords(self._sizes_real)

    @inline(.always)
    def sizes_real(self) -> ref [origin_of(self._sizes_real)] Self.Vec[.int]:
        return self._sizes_real

    @inline(.always)
    def size_fourier[d: Int](self, out size_fourier: Int):
        comptime if d == 0:
            size_fourier = self.sizes_real()[d] | 0b1
        else:
            size_fourier = self.half().size_fourier[d]()

    def sizes_fourier(self, out sizes_fourier: Self.Vec[.int]):
        sizes_fourier = Self.Vec[.int](uninitialized=True)
        comptime for d in range(Self.dim):
            sizes_fourier[d] = self.size_fourier[d]()

    @inline(.always)
    def _pivot[d: Int](self, out pivot: Int):
        """The first image index that maps to a negative frequency."""
        comptime if d == 0:
            pivot = (self.sizes_real()[d] >> 1) + 1
        else:
            pivot = self.half()._pivot[d]()
    
    @inline(.always)
    def fmin[d: Int](self, out fmin: Int):
        fmin = self.half().fmin[d]()

    @inline(.always)
    def fmin(self, out fmin: Self.Vec[.int]):
        """Returns the lower bound (inclusive) on fourier coordinates."""
        fmin = self.half().fmin()

    @inline(.always)
    def fmax[d: Int](self, out fmax: Int):
        fmax = self.half().fmax[d]()

    @inline(.always)
    def fmax(self, out fmax: Self.Vec[.int]):
        """Returns the upper bound (inclusive) on fourier coordinates."""
        fmax = self.half().fmax()

    @inline(.always)
    def f_in_range[d: Int](self, f: Int, out in_range: Bool):
        in_range = self.half().f_in_range[d](f)

    @inline(.always)
    def f_in_range(self, f: Self.Vec[.int]) -> Bool:
        return self.half().f_in_range(f)

    @inline(.always)
    def f2i(self, f: Self.Vec[.int], out i: Self.Vec[.int]):
        """
        Converts fourier coordinates to image coordinates.
        
        Precondition: f is in range
        """

        i = Self.Vec[.int](uninitialized=True)

        comptime for d in range(Self.dim):
            if f[d] < 0:
                i[d] = f[d] + self.size_fourier[d]()
            else:
                i[d] = f[d]
    
    @inline(.always)
    def i2f(self, i: Self.Vec[.int], out f: Self.Vec[.int]):
        """
        Converts image coordinates to fourier coordinates.
        
        Precondition: i is in range
        """

        f = Self.Vec[.int](uninitialized=True)

        comptime for d in range(Self.dim):
            if i[d] >= self._pivot[d]():
                f[d] = i[d] - self.size_fourier[d]()
            else:
                f[d] = i[d]

    def f_norm[dtype: DType](
        self,
        *,
        f: Self.Vec[.int],
        out f_norm: Self.Vec[dtype]
    ):
        f_norm = self.half().f_norm[dtype](f=f)

    def f_norm[dtype: DType](
        self,
        *,
        i: Self.Vec[.int],
        out f_norm: Self.Vec[dtype]
    ):
        f_norm = self.half().f_norm[dtype](i=i)
