
@fieldwise_init
struct Parameter(ImplicitlyCopyable, Movable, Writable):
    var id: Int64
    var name: StaticString

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ")")
    

struct ParameterSet(Copyable, Movable):
    var _dict: Dict[Int64,Parameter]

    fn __init__(out self, params: List[Parameter]) raises:
        self._dict = {}
        for param in params:
            self._add(param)

    fn __init__(out self, *params: Parameter) raises:
        self._dict = {}
        for param in params:
            self._add(param)

    fn _add(mut self, param: Parameter) raises:
        if param.id in self._dict:
            raise Error("Duplicate parameter in set: ", param)
        self._dict[param.id] = param

    fn __getitem__(self, id: Int64) -> Optional[Pointer[Parameter, origin=ImmutableOrigin.cast_from[__origin_of(self._dict.__getitem__(id))]]]:
        try:
            return Pointer(to=self._dict[id])
        except:
            return None


struct CistemParameters:

    alias position_in_stack = Parameter(1 << 0, "position_in_stack")
    alias image_is_active = Parameter(1 << 1, "image_is_active")
    alias psi = Parameter(1 << 2, "psi")
    alias x_shift = Parameter(1 << 3, "x_shift")
    alias y_shift = Parameter(1 << 4, "y_shift")
    alias defocus_1 = Parameter(1 << 5, "defocus_1")
    alias defocus_2 = Parameter(1 << 6, "defocus_2")
    alias defocus_angle = Parameter(1 << 7, "defocus_angle")
    alias phase_shift = Parameter(1 << 8, "phase_shift")
    alias occupancy = Parameter(1 << 9, "occupancy")
    alias logp = Parameter(1 << 10, "logp")
    alias sigma = Parameter(1 << 11, "sigma")
    alias score = Parameter(1 << 12, "score")
    alias score_change = Parameter(1 << 13, "score_change")
    alias pixel_size = Parameter(1 << 14, "pixel_size")
    alias microscope_voltage = Parameter(1 << 15, "microscope_voltage")
    alias microscope_cs = Parameter(1 << 16, "microscope_cs")
    alias amplitude_contrast = Parameter(1 << 17, "amplitude_contrast")
    alias beam_tilt_x = Parameter(1 << 18, "beam_tilt_x")
    alias beam_tilt_y = Parameter(1 << 19, "beam_tilt_y")
    alias image_shift_x = Parameter(1 << 20, "image_shift_x")
    alias image_shift_y = Parameter(1 << 21, "image_shift_y")
    alias theta = Parameter(1 << 22, "theta")
    alias phi = Parameter(1 << 23, "phi")
    alias stack_filename = Parameter(1 << 24, "stack_filename")
    alias original_image_filename = Parameter(1 << 25, "original_image_filename")
    alias reference_3d_filename = Parameter(1 << 26, "reference_3d_filename")
    alias best_2d_class = Parameter(1 << 27, "best_2d_class")
    alias beam_tilt_group = Parameter(1 << 28, "beam_tilt_group")
    alias particle_group = Parameter(1 << 29, "particle_group")
    alias pre_exposure = Parameter(1 << 30, "pre_exposure")
    alias total_exposure = Parameter(1 << 31, "total_exposure")
    alias assigned_subset = Parameter(1 << 32, "assigned_subset")
    alias original_x_position = Parameter(1 << 33, "original_x_position")
    alias original_y_position = Parameter(1 << 34, "original_y_position")

    alias _all = List[Parameter](
        Self.position_in_stack,
        Self.image_is_active,
        Self.psi,
        Self.x_shift,
        Self.y_shift,
        Self.defocus_1,
        Self.defocus_2,
        Self.defocus_angle,
        Self.phase_shift,
        Self.occupancy,
        Self.logp,
        Self.sigma,
        Self.score,
        Self.score_change,
        Self.pixel_size,
        Self.microscope_voltage,
        Self.microscope_cs,
        Self.amplitude_contrast,
        Self.beam_tilt_x,
        Self.beam_tilt_y,
        Self.image_shift_x,
        Self.image_shift_y,
        Self.theta,
        Self.phi,
        Self.stack_filename,
        Self.original_image_filename,
        Self.reference_3d_filename,
        Self.best_2d_class,
        Self.beam_tilt_group,
        Self.particle_group,
        Self.pre_exposure,
        Self.total_exposure,
        Self.assigned_subset,
        Self.original_x_position,
        Self.original_y_position
    )

    @staticmethod
    fn all() -> List[Parameter]:
        return materialize[Self._all]()
