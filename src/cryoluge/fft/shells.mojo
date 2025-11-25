
from math import sqrt

from cryoluge.math import Dimension, Vec


@fieldwise_init
struct FourierShells[dim: Dimension](
    Copyable,
    Movable
):
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
