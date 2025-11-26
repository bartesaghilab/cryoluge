
from math import sqrt

from cryoluge.math import Dimension, Vec
from cryoluge.fft import FFTCoords


@fieldwise_init
struct FourierShells[dim: Dimension](
    Copyable,
    Movable,
    Sized
):
    """
    Calculates fourier shell indices based on frequency coordinates.
    When the count is the largest dimension of an area, the index is in the range:
        [ 0, count*sqrt(2)/2 ) or about 0.707
    When the count is the largest dimension of a volume, the index is in the range:
        [ 0, count*sqrt(3)/2 ) or about 0.866
    Therefore, the shell index can never exceed the count for areas and volumes.
    """

    var sizes_real: Vec[Int,dim]
    var count: Int

    fn shelli[dtype: DType](
        self,
        *,
        freq2: Scalar[dtype],
        out shelli: Int
    ):
        """Returns the index of the Fourier shell at the given squared frequency."""
        shelli = Int(sqrt(freq2)*self.count)
    
    fn shelli[dtype: DType](
        self,
        *,
        f: Vec[Int,dim],
        out shelli: Int
    ):
        """Returns the index of the Fourier shell at the given frequency coordinates."""
        var freqs = FFTCoords(self.sizes_real).freqs[dtype](f=f)
        shelli = self.shelli(freq2=freqs.len2())

    fn shelli_max(self, out result: Int):
        var freq_max = sqrt(Scalar[DType.float32](dim.rank))
        result = Int(freq_max*self.count/2)

    fn __len__(self) -> Int:
        return self.shelli_max() + 1
