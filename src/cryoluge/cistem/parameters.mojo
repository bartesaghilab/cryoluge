
from sys import size_of

from cryoluge.collections import Keyable


@fieldwise_init
struct Parameter(ImplicitlyCopyable, Movable, Writable, Stringable, EqualityComparable, Keyable):
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

    comptime Key = Int64
    fn key(self) -> Self.Key:
        return self.id


@fieldwise_init
struct ParameterType(ImplicitlyCopyable, Movable, Writable, Stringable, EqualityComparable):
    var id: UInt8
    var name: StaticString
    var dtype: Optional[DType]
    var size: Optional[Int]

    comptime text = Self(1, "text")
    comptime int = Self.of[DType.int32](2, "int")
    comptime float = Self.of[DType.float32](3, "float")
    comptime bool = Self.of[DType.bool](4, "bool")
    comptime long = Self.of[DType.int64](5, "long")
    comptime double = Self.of[DType.float64](6, "double")
    comptime byte = Self.of[DType.uint8](7, "byte")
    comptime vary = Self(8, "vary")
    comptime uint = Self.of[DType.uint32](9, "uint")

    comptime all = List[Self](
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

    fn __init__(out self, id: UInt8, name: StaticString):
        self = Self(id, name, None, None)

    @staticmethod
    fn of[dtype: DType](id: UInt8, name: StaticString, out self: Self):
        self = Self(id, name, dtype, size_of[dtype]())

    @staticmethod
    fn get(id: UInt8) -> Optional[Self]:
        @parameter
        for t in Self.all:
            if t.id == id:
                return t
        return None

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id


struct CistemParameters:

    comptime position_in_stack = Parameter(1 << 0, "position_in_stack", ParameterType.uint)
    comptime image_is_active = Parameter(1 << 1, "image_is_active", ParameterType.int)
    comptime psi = Parameter(1 << 2, "psi", ParameterType.float)
    comptime x_shift = Parameter(1 << 3, "x_shift", ParameterType.float)
    comptime y_shift = Parameter(1 << 4, "y_shift", ParameterType.float)
    comptime defocus_1 = Parameter(1 << 5, "defocus_1", ParameterType.float)
    comptime defocus_2 = Parameter(1 << 6, "defocus_2", ParameterType.float)
    comptime defocus_angle = Parameter(1 << 7, "defocus_angle", ParameterType.float)
    comptime phase_shift = Parameter(1 << 8, "phase_shift", ParameterType.float)
    comptime occupancy = Parameter(1 << 9, "occupancy", ParameterType.float)
    comptime logp = Parameter(1 << 10, "logp", ParameterType.float)
    comptime sigma = Parameter(1 << 11, "sigma", ParameterType.float)
    comptime score = Parameter(1 << 12, "score", ParameterType.float)
    comptime score_change = Parameter(1 << 13, "score_change", ParameterType.float)
    comptime pixel_size = Parameter(1 << 14, "pixel_size", ParameterType.float)
    comptime microscope_voltage = Parameter(1 << 15, "microscope_voltage", ParameterType.float)
    comptime microscope_cs = Parameter(1 << 16, "microscope_cs", ParameterType.float)
    comptime amplitude_contrast = Parameter(1 << 17, "amplitude_contrast", ParameterType.float)
    comptime beam_tilt_x = Parameter(1 << 18, "beam_tilt_x", ParameterType.float)
    comptime beam_tilt_y = Parameter(1 << 19, "beam_tilt_y", ParameterType.float)
    comptime image_shift_x = Parameter(1 << 20, "image_shift_x", ParameterType.float)
    comptime image_shift_y = Parameter(1 << 21, "image_shift_y", ParameterType.float)
    comptime theta = Parameter(1 << 22, "theta", ParameterType.float)
    comptime phi = Parameter(1 << 23, "phi", ParameterType.float)
    comptime stack_filename = Parameter(1 << 24, "stack_filename", ParameterType.text)
    comptime original_image_filename = Parameter(1 << 25, "original_image_filename", ParameterType.text)
    comptime reference_3d_filename = Parameter(1 << 26, "reference_3d_filename", ParameterType.text)
    comptime best_2d_class = Parameter(1 << 27, "best_2d_class", ParameterType.int)
    comptime beam_tilt_group = Parameter(1 << 28, "beam_tilt_group", ParameterType.int)
    comptime particle_group = Parameter(1 << 29, "particle_group", ParameterType.int)
    comptime pre_exposure = Parameter(1 << 30, "pre_exposure", ParameterType.float)
    comptime total_exposure = Parameter(1 << 31, "total_exposure", ParameterType.float)
    comptime assigned_subset = Parameter(1 << 32, "assigned_subset", ParameterType.int)
    comptime original_x_position = Parameter(1 << 33, "original_x_position", ParameterType.float)
    comptime original_y_position = Parameter(1 << 34, "original_y_position", ParameterType.float)

    comptime _all = List[Parameter](
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
