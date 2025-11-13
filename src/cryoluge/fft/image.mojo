
from math import pi, cos, sin, floor

from cryoluge.math import Dimension, Vec, ComplexScalar, Matrix
from cryoluge.image import ComplexImage


struct FFTImage[
    dim: Dimension,
    dtype: DType
](Copyable, Movable):
    """
    A thin wrapper around ComplexImage that remembers the real sizes,
    to enable coordinate transfomrations between real-space and fourier-space.
    """

    var sizes_real: Vec[Int,dim]
    var complex: ComplexImage[dim,dtype]

    alias D1 = FFTImage[Dimension.D1,_]
    alias D2 = FFTImage[Dimension.D2,_]
    alias D3 = FFTImage[Dimension.D3,_]
    alias Vec = ComplexImage[dim,dtype].Vec
    alias PixelType = ComplexImage[dim,dtype].PixelType
    alias PixelVec = ComplexImage[dim,dtype].PixelVec
    alias ScalarType = ComplexImage[dim,dtype].ScalarType
    alias ScalarVec = ComplexImage[dim,dtype].ScalarVec

    fn __init__(out self, sizes_real: Self.Vec[Int], *, alignment: Optional[Int] = None):
        self.sizes_real = sizes_real.copy()
        var fft_coords = FFTCoords(sizes_real)
        self.complex = ComplexImage[dim,dtype](fft_coords.sizes_fourier(), alignment=alignment)

    fn __init__(out self, *, of: Image[dim,dtype], alignment: Optional[Int] = None):
        ref real = of
        self = Self(real.sizes(), alignment=alignment)

    fn coords(self) -> FFTCoords[dim, origin=__origin_of(self.sizes_real)]:
        return FFTCoords(self.sizes_real)

    fn crop(self, *, mut to: Self):
        ref dst = to

        # make sure the destination image is smaller (or the same size) as this one
        @parameter
        for d in range(dim.rank):
            debug_assert(
                dst.sizes_real[d] <= self.sizes_real[d],
                "Crop destination real sizes ", dst.sizes_real,
                " must be smaller (or same size) as this source real sizes ", self.sizes_real
            )

        # sample into the dst image
        @parameter
        fn sample(i: Self.Vec[Int]):
            dst.complex[i] = self.complex[self.coords().f2i(dst.coords().i2f(i))]

        dst.complex.iterate[sample]()

    # TODO: separate from iterate()
    fn phase_shift(mut self, shift: Self.Vec[Scalar[dtype]]):
        
        @parameter
        fn func(i: Self.Vec[Int]):
            var freq = self.coords().i2f(i).map_scalar[dtype]()
            var sizes_real = self.coords().sizes_real().map_scalar[dtype]()
            var phase = 0 - (freq*shift*2*pi/sizes_real).sum()
            self.complex[i=i] *= ComplexScalar[dtype](re=cos(phase), im=sin(phase))

        self.complex.iterate[func]()

    fn get(
        self,
        *,
        f: Self.Vec[Int],
        out v: Optional[ComplexScalar[dtype]]
    ):
        var conj = self.coords().needs_conjugation(f=f)
        var i = self.coords().maybe_f2i(f, needs_conj=conj)

        if i is None:
            v = None
            return

        v = self.complex.get(i.value())
        if v is not None and conj:
            v = v.value().conj()

    fn get(
        self: FFTImage[dim,dtype],
        *,
        f_lerp: Self.Vec[Float32]
    ) -> Optional[ComplexScalar[dtype]]:
        ref f = f_lerp

        # discretize the frequency coordinates
        @parameter
        fn func_start(v: Float32) -> Int:
            return Int(floor(v))
        var start = f.map[mapper=func_start]()

        # build the multi-dimensional delta vectors (at compile-time)
        fn build_deltas(out deltas: List[Vec[Int,dim]]):
            deltas = [
                Vec[Int,dim](fill=0)
            ]
            for d in range(dim.rank):
                for i in range(len(deltas)):
                    var delta = deltas[i].copy()
                    delta[d] = 1
                    deltas.append(delta^)

        sum = ComplexScalar[dtype](0, 0)

        @parameter
        for delta in build_deltas():
            var f_sample = start + materialize[delta]()

            # TEMP: technically, the sample might need conjugation,
            #       and the coordinate inversion might put it back in-range,
            #       but csp1 doesn't do that,
            #       so add another explicit (and unecessary) range check here to match csp1 behavior
            if not self.coords().f_in_range(f_sample):
                return None
            
            var v = self.get(f=f_sample)
            if v is None:
                return None
            var dists = (f - f_sample.map_float32()).abs()
            var weight = Scalar[dtype]((1 - dists).product())
            sum = sum + v.value()*weight

        return sum

    # TODO: separate from iterate()
    fn slice(
        self: FFTImage[Dimension.D3,dtype],
        *,
        rot: Matrix.D3[DType.float32],
        res_limit: Float32,
        mut to: FFTImage.D2[dtype],
        out_of_range: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0),
        origin_value: ComplexScalar[dtype] = ComplexScalar[dtype](0, 0)
    ):
        var res_limit2 = (res_limit*self.coords().sizes_real().x())**2

        print('hack', res_limit2)  # TEMP: needed to work around compiler bug

        @parameter
        fn func(i: to.Vec[Int]):
            var freq_2d = to.coords().i2f(i)
            var freq_2d_f = freq_2d.map_float32()

            if freq_2d_f.len2() <= res_limit2:

                # rotate the sample point into 3d
                var freq_3d_f = rot*freq_2d_f.lift(z=0)

                # do the linear interpolation
                var v = self.get(f_lerp=freq_3d_f).or_else(out_of_range)

                # save to the output
                if to.coords().needs_conjugation(f=freq_2d):
                    v = v.conj()
                to.complex[i] = v
            else:
                to.complex[i] = out_of_range

        to.complex.iterate[func]()

        # set origin to control the average
        to.complex[x=0, y=0] = origin_value
