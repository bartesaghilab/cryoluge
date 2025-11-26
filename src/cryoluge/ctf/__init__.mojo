
from math import sqrt, atan, atan2
from complex import ComplexScalar

from cryoluge.math import Vec
from cryoluge.math.units import UnitType, Unit, Px, Ang, MM, Rad, Deg, pi, Hz
from cryoluge.fft import FFTCoords


struct CTF[dtype: DType](
    Copyable,
    Movable
):
    var wavelength_px: Px[dtype]
    var wavelength2_px: Px[dtype]
    var spherical_aberration_px: Px[dtype]
    var _defocus_px: Vec.D2[Px[dtype]]
    var astigmatism_azimuth_rad: Rad[dtype]
    var pixel_size: Scalar[dtype]
    var additional_phase_shift_rad: Rad[dtype]
    var amplitude_contrast_term_rad: Rad[dtype]
    var _beam_tilt_rad: Rad[dtype]
    var beam_tilt_azimuth_rad: Rad[dtype]
    var _particle_shift_px: Px[dtype]
    var particle_shift_azimuth_rad: Rad[dtype]
    var low_resolution_contrast: Scalar[dtype]  # TODO: does this have a unit?

    fn __init__(
        out self,
        *,
        acceleration_voltage_kv: Scalar[dtype],
        spherical_aberration_mm: MM[dtype],
        amplitude_contrast: Scalar[dtype],  # TODO: does this have a unit?
        defocus_a: Vec.D2[Ang[dtype]],
        astigmatism_azimuth_rad: Rad[dtype],
        pixel_size: Scalar[dtype],
        additional_phase_shift_rad: Rad[dtype],
        beam_tilt_rad: Vec.D2[Rad[dtype]],
        particle_shift_a: Vec.D2[Ang[dtype]],
        low_resolution_contrast: Scalar[dtype] = 0
    ):
        self.wavelength_px = _acceleration_voltage_to_wavelength_a(acceleration_voltage_kv).to_px(pixel_size)
        self.wavelength2_px = self.wavelength_px**2
        self.spherical_aberration_px = spherical_aberration_mm.to_ang().to_px(pixel_size)

        @parameter
        fn to_px(a: Ang[dtype]) -> Px[dtype]:
            return a.to_px(pixel_size)
        self._defocus_px = defocus_a.map[mapper=to_px]()

        self.astigmatism_azimuth_rad = astigmatism_azimuth_rad
        self.pixel_size = pixel_size
        self.additional_phase_shift_rad = additional_phase_shift_rad

        # handle amplitude contrast
        if abs(amplitude_contrast - 1) < 1e-3:
            self.amplitude_contrast_term_rad = pi[dtype]/2
        else:
            self.amplitude_contrast_term_rad = Rad(atan(amplitude_contrast / sqrt(1 - amplitude_contrast**2)))

        # handle beam tilt
        self._beam_tilt_rad = beam_tilt_rad.len()
        if beam_tilt_rad.x() < 1e-4 and beam_tilt_rad.y() < 1e-4:
            self.beam_tilt_azimuth_rad = Rad[dtype](0)
        else:
            self.beam_tilt_azimuth_rad = Rad(atan2(beam_tilt_rad.y().value, beam_tilt_rad.x().value))

        # handle particle shift
        var particle_shift_px = particle_shift_a.map[mapper=to_px]()
        self._particle_shift_px = particle_shift_px.len()
        if particle_shift_px.x() < 1e-4 and particle_shift_px.y() < 1e-4:
            self.particle_shift_azimuth_rad = Rad[dtype](0)
        else:
            self.particle_shift_azimuth_rad = Rad(atan2(particle_shift_px.x().value, particle_shift_px.y().value))

        self.low_resolution_contrast = low_resolution_contrast

    fn defocus_px(self) -> ref [origin_of(self._defocus_px)] Vec.D2[Px[dtype]]:
        return self._defocus_px

    fn defocus_px(self, *, azimuth_rad: Rad[dtype], out defocus_px: Px[dtype]):
        var sum = self._defocus_px[0] + self._defocus_px[1]
        var diff = self._defocus_px[0] - self._defocus_px[1]
        defocus_px = 0.5*( sum + diff*(2*(azimuth_rad - self.astigmatism_azimuth_rad)).cos() )

    fn phase_shift_rad(
        self,
        *,
        spatial_freq2_hz: Hz[dtype],
        azimuth_rad: Rad[dtype],
        out phase_shift_rad: Rad[dtype]
    ):
        var px = self.wavelength_px*(
            self.defocus_px(azimuth_rad=azimuth_rad)
            - 0.5*self.wavelength2_px*spatial_freq2_hz.value*self.spherical_aberration_px
        )
        phase_shift_rad = pi[dtype]*spatial_freq2_hz.value*px.value
            + self.additional_phase_shift_rad
            + self.amplitude_contrast_term_rad

    fn phase_shift_rad(
        self,
        *,
        spatial_freq2_hz: Hz[dtype],
        beam_tilt_rad: Rad[dtype],
        particle_shift_px: Px[dtype],
        out phase_shift_rad: Rad[dtype]
    ):
        var spatial_freq_hz = spatial_freq2_hz.sqrt()
        phase_shift_rad = 2*pi[dtype]*(
            beam_tilt_rad*(
                (self.spherical_aberration_px*self.wavelength2_px).value
                * (spatial_freq2_hz*spatial_freq_hz).value
            )
            - spatial_freq_hz.value*particle_shift_px.value
        ).normalize_minus_pi_to_pi()

    fn beam_tilt_rad(self, out beam_tilt_rad: Rad[dtype]):
        beam_tilt_rad = self._beam_tilt_rad

    fn beam_tilt_rad(
        self,
        *,
        azimuth_rad: Rad[dtype],
        out beam_tilt_rad: Rad[dtype]
    ):
        beam_tilt_rad = self._beam_tilt_rad*(azimuth_rad - self.beam_tilt_azimuth_rad).cos()

    fn particle_shift_px(
        self,
        *,
        azimuth_rad: Rad[dtype],
        out particle_shift_px: Px[dtype]
    ):
        particle_shift_px = self._particle_shift_px*(azimuth_rad - self.particle_shift_azimuth_rad).cos()

    fn eval(
        self,
        *,
        spatial_freq2_hz: Hz[dtype],
        azimuth_rad: Rad[dtype]
    ) -> Scalar[dtype]:

        # TODO: this seems pretty arbitrary?
        if self._defocus_px[0].value == 0 and self._defocus_px[1].value == 0:
            return -0.7  # for defocus sweep

        comptime threshold = 2*pi[dtype]

        var phase_shift_rad = self.phase_shift_rad(spatial_freq2_hz=spatial_freq2_hz, azimuth_rad=azimuth_rad)
        if self.low_resolution_contrast == 0 or phase_shift_rad >= threshold:
            return -phase_shift_rad.sin()

        return -(phase_shift_rad + self.low_resolution_contrast*(1 - phase_shift_rad/threshold)).sin()

    fn eval(
        self,
        *,
        dist: Vec.D2[Scalar[dtype]],
        out result: Scalar[dtype]
    ):
        result = self.eval(
            spatial_freq2_hz=Hz(dist.len2()),
            azimuth_rad=_azimuth_rad(dist)
        )

    fn eval(
        self,
        *,
        f: Vec.D2[Int],
        sizes_real: Vec.D2[Int],
        out result: Scalar[dtype]
    ):
        result = self.eval(
            dist=FFTCoords(sizes_real).freqs[dtype](f=f)
        )

    fn eval(
        self,
        *,
        i: Vec.D2[Int],
        sizes_real: Vec.D2[Int],
        out result: Scalar[dtype]
    ):
        result = self.eval(
            f=FFTCoords(sizes_real).i2f(i),
            sizes_real=sizes_real
        )

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        spatial_freq2_hz: Hz[dtype],
        azimuth_rad: Rad[dtype],
        out result: ComplexScalar[dtype]
    ):
        if self._beam_tilt_rad.value == 0 and self._particle_shift_px.value == 0:
            result = ComplexScalar[dtype](1, 0)

        var phase_shift_rad = self.phase_shift_rad(
            spatial_freq2_hz=spatial_freq2_hz,
            beam_tilt_rad=self.beam_tilt_rad(azimuth_rad=azimuth_rad),
            particle_shift_px=self.particle_shift_px(azimuth_rad=azimuth_rad)
        )
        result = ComplexScalar[dtype](phase_shift_rad.cos(), phase_shift_rad.sin())

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        dist: Vec.D2[Scalar[dtype]],
        out result: ComplexScalar[dtype]
    ):
        result = self.eval_beam_tilt_phase_shift(
            spatial_freq2_hz=Hz(dist.len2()),
            azimuth_rad=_azimuth_rad(dist)
        )

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        f: Vec.D2[Int],
        sizes_real: Vec.D2[Int],
        out result: ComplexScalar[dtype]
    ):
        result = self.eval_beam_tilt_phase_shift(
            dist=FFTCoords(sizes_real).freqs[dtype](f=f)
        )

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        i: Vec.D2[Int],
        sizes_real: Vec.D2[Int],
        out result: ComplexScalar[dtype]
    ):
        result = self.eval_beam_tilt_phase_shift(
            f=FFTCoords(sizes_real).i2f(i),
            sizes_real=sizes_real
        )


fn _acceleration_voltage_to_wavelength_a[dtype: DType](
    acceleration_voltage_kv: Scalar[dtype],
    out wavelength_a: Ang[dtype]
):
    var acceleration_voltage_v = 1000*acceleration_voltage_kv
    wavelength_a = Ang(12.2639 / sqrt(acceleration_voltage_v + 0.97845e-6*acceleration_voltage_v**2))


fn _azimuth_rad[dtype: DType](dist: Vec.D2[Scalar[dtype]], out azimuth_rad: Rad[dtype]):
    azimuth_rad = Rad[dtype](0)
    if dist.x() != 0 or dist.y() != 0:
        azimuth_rad = Rad(atan2(dist.y(), dist.x()))
