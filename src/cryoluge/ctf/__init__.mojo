
from math import sqrt, pi, atan, atan2

from cryoluge.math import Vec
from cryoluge.math.units import UnitType, Unit, Pix, Ang, MM, Rad, Deg


struct CTF[dtype: DType](
    Copyable,
    Movable
):
    var wavelength_px: Pix[dtype]
    var wavelength2_px: Pix[dtype]
    var spherical_aberration_px: Pix[dtype]
    var defocus_px: Vec.D2[Pix[dtype]]
    var astigmatism_azimuth_rad: Rad[dtype]
    var additional_phase_shift_rad: Rad[dtype]
    var amplitude_contrast: Scalar[dtype]  # TODO: does this have a unit?
    var precomputed_amplitude_contrast_term_rad: Rad[dtype]
    var beam_tilt_rad: Rad[dtype]
    var beam_tilt_azimuth_rad: Rad[dtype]
    var particle_shift_px: Pix[dtype]
    var particle_shift_azimuth_rad: Rad[dtype]

    fn __init__(
        out self,
        *,
        acceleration_voltage_kv: Scalar[dtype],
        spherical_aberration_mm: MM[dtype],
        amplitude_contrast: Scalar[dtype],
        defocus_a: Vec.D2[Ang[dtype]],
        astigmatism_azimuth_deg: Deg[dtype],
        pixel_size: Scalar[dtype],
        additional_phase_shift_rad: Rad[dtype],
        beam_tilt_rad: Vec.D2[Rad[dtype]],
        particle_shift_a: Vec.D2[Rad[dtype]]
    ):

        # TODO: always zero? could be inlined?
        var low_resolution_contrast = 0

        self.wavelength_px = _acceleration_voltage_to_wavelength_a(acceleration_voltage_kv).to_pix(pixel_size)
        self.wavelength2_px = self.wavelength_px**2
        self.spherical_aberration_px = spherical_aberration_mm.to_ang().to_pix(pixel_size)

        @parameter
        fn to_pix(a: Ang[dtype]) -> Pix[dtype]:
            return a.to_pix(pixel_size)
        self.defocus_px = defocus_a.map[mapper=to_pix]()

        self.astigmatism_azimuth_rad = astigmatism_azimuth_deg.to_rad()
        self.additional_phase_shift_rad = additional_phase_shift_rad

        # handle amplitude contrast
        self.amplitude_contrast = amplitude_contrast
        if abs(self.amplitude_contrast - 1) < 1e-3:
            self.precomputed_amplitude_contrast_term_rad = Rad[dtype](pi/2)
        else:
            self.precomputed_amplitude_contrast_term_rad = Rad(atan(self.amplitude_contrast / sqrt(1 - self.amplitude_contrast**2)))

        # handle beam tilt
        self.beam_tilt_rad = beam_tilt_rad.len()
        if beam_tilt_rad.x() < 1e-4 and beam_tilt_rad.y() < 1e-4:
            self.beam_tilt_azimuth_rad = Rad[dtype](0)
        else:
            self.beam_tilt_azimuth_rad = Rad(atan2(beam_tilt_rad.y().value, beam_tilt_rad.x().value))

        # handle particle shift
        var particle_shift_px = particle_shift_a.map[mapper=to_pix]()
        self.particle_shift_px = particle_shift_px.len()
        if particle_shift_px.x() < 1e-4 and particle_shift_px.y() < 1e-4:
            self.particle_shift_azimuth_rad = Rad[dtype](0)
        else:
            self.particle_shift_azimuth_rad = Rad(atan2(particle_shift_px.x().value, particle_shift_px.y().value))


fn _acceleration_voltage_to_wavelength_a[dtype: DType](
    acceleration_voltage_kv: Scalar[dtype],
    out wavelength_a: Ang[dtype]
):
    var acceleration_voltage_v = 1000*acceleration_voltage_kv
    wavelength_a = Ang(12.2639 / sqrt(acceleration_voltage_v + 0.97845e-6*acceleration_voltage_v**2))
