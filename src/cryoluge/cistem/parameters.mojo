
@fieldwise_init
struct Parameter(ImplicitlyCopyable, Movable, Writable, Stringable, EqualityComparable):
    var id: Int64
    var name: StaticString
    var type: ParameterType

    @staticmethod
    fn unknown(id: Int64, type: ParameterType) -> Self:
        return Self(id, "?", type)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ",", self.type, ")")

    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id
            and self.name == other.name
            and self.type == other.type


@fieldwise_init
struct ParameterType(ImplicitlyCopyable, Movable, Writable, Stringable, EqualityComparable):
    var id: UInt8
    var name: StaticString
    var dtype: Optional[DType]

    alias text = Self(1, "text", None)
    alias int = Self(2, "int", DType.int32)
    alias float = Self(3, "float", DType.float32)
    alias bool = Self(4, "bool", DType.bool)
    alias long = Self(5, "long", DType.int64)
    alias double = Self(6, "double", DType.float64)
    alias byte = Self(7, "byte", DType.uint8)
    alias vary = Self(8, "vary", None)
    alias uint = Self(9, "uint", DType.uint32)

    alias all = List[Self](
        Self.text,
        Self.int,
        Self.float,
        Self.bool,
        Self.long,
        Self.double,
        Self.byte,
        Self.vary,
        Self.uint
    )

    @staticmethod
    fn get(id: UInt8) raises -> Self:
        @parameter
        for t in Self.all:
            if t.id == id:
                return t
        raise Error(String("Unrecognized column type id: ", id)) 

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id


struct ParameterSet(Copyable, Movable):
    var _dict: Dict[Int64,Parameter]

    # TODO: can we use Iterator trait here?
    fn __init__(out self, params: List[Parameter]) raises:
        self._dict = {}
        for param in params:
            self._add(param)

    # TODO: Iterable in fn arg pos not stable yet
    #       see: https://forum.modular.com/t/iterable-trait-as-function-argument/2284/2
    # fn __init__[I: Iterable](out self, params: I) raises:
    #     self._dict = {}
    #     for param in params:
    #         # can't express bounds on trait associated aliases, yet, so need to rebind iterator value
    #         ref p = rebind[Parameter](param)
    #         self._add(p)

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

    alias position_in_stack = Parameter(1 << 0, "position_in_stack", ParameterType.uint)
    alias image_is_active = Parameter(1 << 1, "image_is_active", ParameterType.int)
    alias psi = Parameter(1 << 2, "psi", ParameterType.float)
    alias x_shift = Parameter(1 << 3, "x_shift", ParameterType.float)
    alias y_shift = Parameter(1 << 4, "y_shift", ParameterType.float)
    alias defocus_1 = Parameter(1 << 5, "defocus_1", ParameterType.float)
    alias defocus_2 = Parameter(1 << 6, "defocus_2", ParameterType.float)
    alias defocus_angle = Parameter(1 << 7, "defocus_angle", ParameterType.float)
    alias phase_shift = Parameter(1 << 8, "phase_shift", ParameterType.float)
    alias occupancy = Parameter(1 << 9, "occupancy", ParameterType.float)
    alias logp = Parameter(1 << 10, "logp", ParameterType.float)
    alias sigma = Parameter(1 << 11, "sigma", ParameterType.float)
    alias score = Parameter(1 << 12, "score", ParameterType.float)
    alias score_change = Parameter(1 << 13, "score_change", ParameterType.float)
    alias pixel_size = Parameter(1 << 14, "pixel_size", ParameterType.float)
    alias microscope_voltage = Parameter(1 << 15, "microscope_voltage", ParameterType.float)
    alias microscope_cs = Parameter(1 << 16, "microscope_cs", ParameterType.float)
    alias amplitude_contrast = Parameter(1 << 17, "amplitude_contrast", ParameterType.float)
    alias beam_tilt_x = Parameter(1 << 18, "beam_tilt_x", ParameterType.float)
    alias beam_tilt_y = Parameter(1 << 19, "beam_tilt_y", ParameterType.float)
    alias image_shift_x = Parameter(1 << 20, "image_shift_x", ParameterType.float)
    alias image_shift_y = Parameter(1 << 21, "image_shift_y", ParameterType.float)
    alias theta = Parameter(1 << 22, "theta", ParameterType.float)
    alias phi = Parameter(1 << 23, "phi", ParameterType.float)
    alias stack_filename = Parameter(1 << 24, "stack_filename", ParameterType.text)
    alias original_image_filename = Parameter(1 << 25, "original_image_filename", ParameterType.text)
    alias reference_3d_filename = Parameter(1 << 26, "reference_3d_filename", ParameterType.text)
    alias best_2d_class = Parameter(1 << 27, "best_2d_class", ParameterType.int)
    alias beam_tilt_group = Parameter(1 << 28, "beam_tilt_group", ParameterType.int)
    alias particle_group = Parameter(1 << 29, "particle_group", ParameterType.int)
    alias pre_exposure = Parameter(1 << 30, "pre_exposure", ParameterType.float)
    alias total_exposure = Parameter(1 << 31, "total_exposure", ParameterType.float)
    alias assigned_subset = Parameter(1 << 32, "assigned_subset", ParameterType.int)
    alias original_x_position = Parameter(1 << 33, "original_x_position", ParameterType.float)
    alias original_y_position = Parameter(1 << 34, "original_y_position", ParameterType.float)

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
