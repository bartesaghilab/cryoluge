
from cryoluge.math import Dimension, Vec, is_odd


struct FFTCoords[
    dim: Dimension,
    origin: Origin[mut=False]
](
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

    var _sizes_real: Pointer[Self.Vec, origin]

    comptime Vec = Vec[Int,dim]

    fn __init__(out self, ref [origin] sizes_real: Self.Vec):
        self._sizes_real = Pointer(to=sizes_real)

    @always_inline
    fn sizes_real(self) -> ref [origin] Self.Vec:
        return self._sizes_real[]

    @always_inline
    fn size_fourier[d: Int](self, out size_fourier: Int):
        @parameter
        if d == 0:
            size_fourier = (self.sizes_real()[d] >> 1) + 1
        else:
            size_fourier = self.sizes_real()[d]

    fn sizes_fourier(self, out sizes_fourier: Self.Vec):
        sizes_fourier = Self.Vec(uninitialized=True)
        @parameter
        for d in range(dim.rank):
            sizes_fourier[d] = self.size_fourier[d]()

    @always_inline
    fn _pivot[d: Int](self, out pivot: Int):
        """The first image index that maps to a negative frequency."""
        @parameter
        if d == 0:
            constrained[False, "No pivot for x"]()
            pivot = 0
        else:
            pivot = (self.sizes_real()[d] + 1) >> 1
    
    @always_inline
    fn fmin[d: Int](self, out fmin: Int):
        @parameter
        if d == 0:
            fmin = 1 - self.size_fourier[d]()
        else:
            fmin = self._pivot[d]() - self.sizes_real()[d]

    @always_inline
    fn fmin(self, out fmin: Self.Vec):
        """Returns the lower bound (inclusive) on fourier coordinates."""
        fmin = Self.Vec(uninitialized=True)
        @parameter
        for d in range(dim.rank):
            fmin[d] = self.fmin[d]()

    @always_inline
    fn fmax[d: Int](self, out fmax: Int):
        @parameter
        if d == 0:
            fmax = self.size_fourier[d]() - 1
        else:
            fmax = self._pivot[d]() - 1

    @always_inline
    fn fmax(self, out fmax: Self.Vec):
        """Returns the upper bound (inclusive) on fourier coordinates."""
        fmax = Self.Vec(uninitialized=True)
        @parameter
        for d in range(dim.rank):
            fmax[d] = self.fmax[d]()

    @always_inline
    fn needs_conjugation(self, *, f: Self.Vec) -> Bool:
        return f.x() < 0

    @always_inline
    fn f_in_range[d: Int](self, f: Int, out in_range: Bool):
        in_range = f >= self.fmin[d]() and f <= self.fmax[d]()

    @always_inline
    fn f_in_range(self, f: Self.Vec) -> Bool:
        @parameter
        for d in range(dim.rank):
            if not self.f_in_range[d](f[d]):
                return False
        return True

    @always_inline
    fn f2i(self, f: Self.Vec, out i: Self.Vec):
        """
        Converts fourier coordinates to image coordinates.
        Maps negative x fourier coordinates to the positive side.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.

        Precondition: f is in range
        """

        i = Self.Vec(uninitialized=True)

        @parameter
        for d in range(dim.rank):
            if f[d] < 0:
                i[d] = f[d] + self.size_fourier[d]()
            else:
                i[d] = f[d]

    fn maybe_f2i(self, f: Self.Vec, *, out i: Optional[Self.Vec]):
        """
        If the Fourier coordinates are in-range, this function converts them to image coordinates.
        Otherwise, returns None.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.
        """

        var _f = f.copy()
        if self.needs_conjugation(f=f):
            _f = -f

        if self.f_in_range(_f):
            i = self.f2i(_f)
        else:
            i = None

    @always_inline
    fn i2f(self, i: Self.Vec, out f: Self.Vec):
        """
        Converts image coordinates to fourier coordinates.
        
        Precondition: i is in range
        """

        f = i.copy()

        # no transformation needed for x
        # but apply the transformation to y,z
        @parameter
        for d in range(1, dim.rank):
            if i[d] >= self._pivot[d]():
                f[d] -= self.size_fourier[d]()

    fn freqs[dtype: DType](
        self,
        *,
        f: Vec[Int,dim],
        out freqs: Vec[Scalar[dtype],dim]
    ):
        var f_dt = f.map_scalar[dtype]()
        var sizes_real_dt = self.sizes_real().map_scalar[dtype]()
        # TODO: doing the div is higher precision, but doesn't match csp1,
        #       since the int truncation is extremely sensitive to roundoff error
        # freq2 = (f_dt/sizes_real_dt).len2()
        # TODO: see if pre-calculating the voxel sizes and doing a mult (instead of a div) is significantly faster
        var sizes_voxel = 1/sizes_real_dt
        freqs = f_dt*sizes_voxel

    fn freqs[dtype: DType](
        self,
        *,
        i: Vec[Int,dim],
        out freqs: Vec[Scalar[dtype],dim]
    ):
        freqs = self.freqs[dtype](f=self.i2f(i))


struct FFTCoordsFull[
    dim: Dimension,
    origin: Origin[mut=False]
](
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

    var _sizes_real: Pointer[Self.Vec, origin]

    comptime Vec = Vec[Int,dim]

    fn __init__(out self, ref [origin] sizes_real: Self.Vec):
        self._sizes_real = Pointer(to=sizes_real)

    fn _half(self) -> FFTCoords[dim,origin]:
        return FFTCoords[dim,origin](self._sizes_real[])

    @always_inline
    fn sizes_real(self) -> ref [origin] Self.Vec:
        return self._sizes_real[]

    @always_inline
    fn size_fourier[d: Int](self, out size_fourier: Int):
        @parameter
        if d == 0:
            size_fourier = self.sizes_real()[d] | 0b1
        else:
            size_fourier = self._half().size_fourier[d]()

    fn sizes_fourier(self, out sizes_fourier: Self.Vec):
        sizes_fourier = Self.Vec(uninitialized=True)
        @parameter
        for d in range(dim.rank):
            sizes_fourier[d] = self.size_fourier[d]()

    @always_inline
    fn _pivot[d: Int](self, out pivot: Int):
        """The first image index that maps to a negative frequency."""
        @parameter
        if d == 0:
            pivot = (self.sizes_real()[d] >> 1) + 1
        else:
            pivot = self._half()._pivot[d]()
    
    @always_inline
    fn fmin[d: Int](self, out fmin: Int):
        fmin = self._half().fmin[d]()

    @always_inline
    fn fmin(self, out fmin: Self.Vec):
        """Returns the lower bound (inclusive) on fourier coordinates."""
        fmin = self._half().fmin()

    @always_inline
    fn fmax[d: Int](self, out fmax: Int):
        fmax = self._half().fmax[d]()

    @always_inline
    fn fmax(self, out fmax: Self.Vec):
        """Returns the upper bound (inclusive) on fourier coordinates."""
        fmax = self._half().fmax()

    @always_inline
    fn f_in_range[d: Int](self, f: Int, out in_range: Bool):
        in_range = self._half().f_in_range[d](f)

    @always_inline
    fn f_in_range(self, f: Self.Vec) -> Bool:
        return self._half().f_in_range(f)

    @always_inline
    fn f2i(self, f: Self.Vec, out i: Self.Vec):
        """
        Converts fourier coordinates to image coordinates.
        
        Precondition: f is in range
        """

        i = Self.Vec(uninitialized=True)

        @parameter
        for d in range(dim.rank):
            if f[d] < 0:
                i[d] = f[d] + self.size_fourier[d]()
            else:
                i[d] = f[d]
    
    @always_inline
    fn i2f(self, i: Self.Vec, out f: Self.Vec):
        """
        Converts image coordinates to fourier coordinates.
        
        Precondition: i is in range
        """

        f = Self.Vec(uninitialized=True)

        @parameter
        for d in range(0, dim.rank):
            if i[d] >= self._pivot[d]():
                f[d] = i[d] - self.size_fourier[d]()
            else:
                f[d] = i[d]

    fn freqs[dtype: DType](
        self,
        *,
        f: Vec[Int,dim],
        out freqs: Vec[Scalar[dtype],dim]
    ):
        freqs = self._half().freqs[dtype](f=f)

    fn freqs[dtype: DType](
        self,
        *,
        i: Vec[Int,dim],
        out freqs: Vec[Scalar[dtype],dim]
    ):
        freqs = self._half().freqs[dtype](i=i)
