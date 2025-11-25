
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

    Follows the "standard" order: For 1D transforms, where A contains the FFT of n real values:
        A contains n/2+1 elements
        A[0] contains the zero-frequency term
        A[1:n/2] contains the positive-frequency terms, in order of ascending frequency

    In higher dimensions, where A[d] contains the d dimension:
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

    fn sizes_real(self) -> ref [origin] Self.Vec:
        return self._sizes_real[]

    fn sizes_fourier(self, out sizes_fourier: Self.Vec):
        sizes_fourier = self.sizes_real().copy()
        sizes_fourier.x() = self.sizes_real().x()//2 + 1
        # NOTE: all the higher dimensions should stay the same

    fn pivot(self, out v: Self.Vec):
        """The first image index that maps to a negative frequency."""
        v = Self.Vec(uninitialized=True)
        v[0] = 0  # not used: can be an arbitrary value
        @parameter
        for d in range(1, dim.rank):
            v[d] = self.sizes_real()[d]//2
            if is_odd(self.sizes_real()[d]):
                v[d] += 1

    fn fmin(self, out v: Self.Vec):
        """Returns the lower bound (inclusive) on fourier coordinates."""
        v = Self.Vec(uninitialized=True)
        v[0] = 1 - self.sizes_fourier()[0]
        @parameter
        for d in range(1, dim.rank):
            v[d] = self.pivot()[d] - self.sizes_real()[d]

    fn fmax(self, out v: Self.Vec):
        """Returns the upper bound (inclusive) on fourier coordinates."""
        v = Self.Vec(uninitialized=True)
        v[0] = self.sizes_fourier()[0] - 1
        @parameter
        for d in range(1, dim.rank):
            v[d] = self.pivot()[d] - 1

    fn needs_conjugation(self, *, f: Self.Vec) -> Bool:
        return f.x() < 0

    fn f_in_range(self, f: Self.Vec) -> Bool:
        @parameter
        for d in range(dim.rank):
            if f[d] < self.fmin()[d] or f[d] > self.fmax()[d]:
                # TODO: do we need to optimize this?? ^^
                return False
        return True

    fn f2i(self, f: Self.Vec, out i: Self.Vec):
        """
        Converts fourier coordinates to image coordinates.
        Maps negative x fourier coordinates to the positive side.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.

        Precondition: f is in range
        """

        i = f.copy()

        @parameter
        for d in range(dim.rank):
            if i[d] < 0:
                i[d] += self.sizes_fourier()[d]

    fn maybe_f2i(self, f: Self.Vec, *, needs_conj: Bool = False, out i: Optional[Self.Vec]):
        """
        If the Fourier coordinates are in-range, this function converts them to image coordinates.
        Otherwise, returns None.

        Check `needs_conjugation()` to see if the FFT value at these coords can be used as-is,
        or if the value needs to be conjugated.
        """

        var _f = f.copy()
        if needs_conj:
            _f = -f

        if self.f_in_range(_f):
            i = self.f2i(_f)
        else:
            i = None

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
            if i[d] >= self.pivot()[d]:
                f[d] = i[d] - self.sizes_fourier()[d]
        # TODO: is the compiler smart enough to lift the function calls outside of the loop?

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
