
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
    var spherical_aberration_px: Px[dtype]
    var astigmatism_azimuth_rad: Rad[dtype]
    var pixel_size: Scalar[dtype]
    var additional_phase_shift_rad: Rad[dtype]
    var amplitude_contrast_term_rad: Rad[dtype]
    var beam_tilt_azimuth_rad: Rad[dtype]
    var particle_shift_azimuth_rad: Rad[dtype]
    var low_resolution_contrast: Scalar[dtype]  # TODO: does this have a unit?
    var _sawl2: Scalar[dtype]
    var _sawl3: Scalar[dtype]
    var _defocus_half_sum: Scalar[dtype]
    var _defocus_half_diff: Scalar[dtype]
    var _extra_rotation: Rad[dtype]
    var _beam_tilt_rad: Rad[dtype]
    var _particle_shift_px: Scalar[dtype]

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
        @parameter
        fn to_px(a: Ang[dtype]) -> Px[dtype]:
            return a.to_px(pixel_size)

        self.wavelength_px = _acceleration_voltage_to_wavelength_a(acceleration_voltage_kv).to_px(pixel_size)
        self.spherical_aberration_px = spherical_aberration_mm.to_ang().to_px(pixel_size)

        self.astigmatism_azimuth_rad = astigmatism_azimuth_rad
        self.pixel_size = pixel_size
        self.additional_phase_shift_rad = additional_phase_shift_rad

        # handle amplitude contrast
        if abs(amplitude_contrast - 1) < 1e-3:
            self.amplitude_contrast_term_rad = pi[dtype]/2
        else:
            self.amplitude_contrast_term_rad = Rad(atan(amplitude_contrast / sqrt(1 - amplitude_contrast**2)))

        # handle beam tilt
        if beam_tilt_rad.x() < 1e-4 and beam_tilt_rad.y() < 1e-4:
            self.beam_tilt_azimuth_rad = Rad[dtype](0)
        else:
            self.beam_tilt_azimuth_rad = Rad(atan2(beam_tilt_rad.y().value, beam_tilt_rad.x().value))

        # handle particle shift
        var particle_shift_px = particle_shift_a.map[mapper=to_px]()
        if particle_shift_px.x() < 1e-4 and particle_shift_px.y() < 1e-4:
            self.particle_shift_azimuth_rad = Rad[dtype](0)
        else:
            self.particle_shift_azimuth_rad = Rad(atan2(particle_shift_px.x().value, particle_shift_px.y().value))

        self.low_resolution_contrast = low_resolution_contrast

        # pre-compute some other terms and factors

        self._sawl2 = (self.spherical_aberration_px * self.wavelength_px**2).value
        self._sawl3 = self._sawl2*self.wavelength_px.value*pi[dtype].value/2

        var defocus_px = defocus_a.map[mapper=to_px]().map_value()
        self._defocus_half_sum = 0.5*(defocus_px.x() + defocus_px.y()) * self.wavelength_px.value*pi[dtype].value
        self._defocus_half_diff = 0.5*(defocus_px.x() - defocus_px.y()) * self.wavelength_px.value*pi[dtype].value

        self._extra_rotation = self.additional_phase_shift_rad + self.amplitude_contrast_term_rad
        self._beam_tilt_rad = beam_tilt_rad.len() * pi[dtype]*2
        self._particle_shift_px = particle_shift_px.len().value * pi[dtype].value*2

    fn azimuth(
        self,
        *,
        freqs: Vec.D2[Scalar[dtype]],
        out result: Rad[dtype]
    ):
        result = Rad[dtype](0)
        if freqs.x() != 0 or freqs.y() != 0:
            result = Rad(atan2(freqs.y(), freqs.x()))

    fn eval(
        self,
        *,
        freqs: Vec.D2[Scalar[dtype]],
        out result: Scalar[dtype]
    ):
        result = self.eval(
            azimuth=self.azimuth(freqs=freqs),
            freq2=freqs.len2()
        )

    fn eval(
        self,
        *,
        azimuth: Rad[dtype],
        freq2: Scalar[dtype],
        out result: Scalar[dtype]
    ):
        var angle = 2*(azimuth - self.astigmatism_azimuth_rad)
        var phase_shift_rad = self._extra_rotation
            + freq2*(self._defocus_half_sum + self._defocus_half_diff*angle.cos() - self._sawl3*freq2)

        comptime threshold = 2*pi[dtype]
        if phase_shift_rad > threshold:
            phase_shift_rad += self.low_resolution_contrast*(1 - phase_shift_rad/threshold)

        return -phase_shift_rad.sin()

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        freqs: Vec.D2[Scalar[dtype]],
        out result: ComplexScalar[dtype]
    ):
        var freq2 = freqs.len2()
        result = self.eval_beam_tilt_phase_shift(
            azimuth=self.azimuth(freqs=freqs),
            freq2=freq2,
            freq=sqrt(freq2)
        )

    fn eval_beam_tilt_phase_shift(
        self,
        *,
        azimuth: Rad[dtype],
        freq2: Scalar[dtype],
        freq: Scalar[dtype],
        out result: ComplexScalar[dtype]
    ):
        if self._beam_tilt_rad.value == 0 and self._particle_shift_px == 0:
            result = ComplexScalar[dtype](1, 0)

        var beam_tilt_rad = self._beam_tilt_rad*(azimuth - self.beam_tilt_azimuth_rad).cos()
        var particle_shift_px = self._particle_shift_px*(azimuth - self.particle_shift_azimuth_rad).cos()

        var phase_shift_rad = freq*(beam_tilt_rad*self._sawl2*freq2 - particle_shift_px)

        result = ComplexScalar[dtype](phase_shift_rad.cos(), phase_shift_rad.sin())


fn _acceleration_voltage_to_wavelength_a[dtype: DType](
    acceleration_voltage_kv: Scalar[dtype],
    out wavelength_a: Ang[dtype]
):
    var acceleration_voltage_v = 1000*acceleration_voltage_kv
    wavelength_a = Ang(12.2639 / sqrt(acceleration_voltage_v + 0.97845e-6*acceleration_voltage_v**2))
