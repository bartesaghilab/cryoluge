
from math import sqrt

from cryoluge.math import Dimension, Vec
from cryoluge.fft import FFTCoords


struct FourierShells[dim: Dimension](
    Copyable,
    Movable,
    Sized
):
    """
    Calculates Fourier shell indices based on frequency coordinates.

    `count` gives the number of Fourier shells that completely fit within the Fourier space.
    Frequency coordinates are normalized such that the distance from
    the center of the Fourier space to any edge is 0.5.
    The frequency distance to the corner depends on the dimensionality of the space:
        2D: sqrt(2)/2 or about 0.707
        3D: sqrt(3)/2 or about 0.866
    
    The shell index is calculated as a proportion of the number
    of shells at unity, ie, when the frequency dist is 1.
    """

    var count: Int
    """The number of shells to completely fit in the Fourier area or volume."""

    var count_at_unity: Int
    """The number of shells in the frequency range [0,1]."""

    comptime D2 = FourierShells[Dimension.D2]
    comptime D3 = FourierShells[Dimension.D3]

    fn __init__(out self, *, size_real: Int):
        """Creates a number of Fourier shells equal to the Fourier half-width in pixels, including the center."""
        var sizes = Vec.D1(x=size_real)
        var coords = FFTCoords(sizes)
        self = Self(
            count = coords.sizes_fourier().x()
        )

    fn __init__(out self, *, count: Int):
        self.count = count
        self.count_at_unity = 2*(count - 1)

    fn shelli[dtype: DType](
        self,
        *,
        freq: Scalar[dtype],
        out shelli: Int
    ):
        """Returns the index of the Fourier shell at the given frequency."""
        shelli = Int(freq*self.count_at_unity)
    
    fn shelli[dtype: DType](
        self,
        *,
        freq2: Scalar[dtype],
        out shelli: Int
    ):
        """Returns the index of the Fourier shell at the given squared frequency."""
        shelli = self.shelli(freq=sqrt(freq2))

    fn freq_max[dtype: DType](self, out freq_max: Scalar[dtype]):
        """Returns the frequency at the corner of Fourier space farthest from the center."""
        freq_max = sqrt(Scalar[dtype](dim.rank))/2
    
    fn shelli_max(self, out shelli_max: Int):
        """Returns the index of the Fourier shell at the corner of Fourier space farthest from the center."""
        shelli_max = self.shelli(freq=self.freq_max[DType.float32]())

    fn __len__(self) -> Int:
        """
        Returns the number of Fourier shells that can fit, at least partially, in the Fourier space.
        ie, includes shells reaching into the corners of the space.
        """
        return self.shelli_max() + 1
