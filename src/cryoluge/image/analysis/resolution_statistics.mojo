
from std.math import sqrt

from cryoluge.math.units import Ang


# NOTE: resolution files are small enough that they can be
#       entirely buffered with no real performance considerations


struct ResolutionStatistics[dtype: DType]:
    var records: List[ResolutionStatisticsRecord[Self.dtype]]

    def __init__(out self):
        self.records = List[ResolutionStatisticsRecord[Self.dtype]]()

    def read(mut self, *, contents: String) raises:

        self.records.clear()

        # read the file line-by-line
        var linei = 0
        for line in contents.splitlines():
            linei += 1

            # skip blank lines
            if line.byte_length() <= 0:
                continue

            # skip comment lines
            var codepoints = line.codepoints()
            if codepoints.__next__() == Codepoint.ord("C"):
                continue

            # parse the columns
            var cols = [w for w in line.split(" ") if w.byte_length() > 0]

            # skip empty lines
            if len(cols) <= 0:
                continue
                
            if len(cols) < 7:
                raise Error("Too few columns (", len(cols), ") on line ", linei)

            self.records.append(ResolutionStatisticsRecord(
                shell = Scalar[Self.dtype](atof(cols[0])),
                resolution = Ang(Scalar[Self.dtype]( atof(cols[1]) )),
                ring_radius = Ang(Scalar[Self.dtype]( atof(cols[2]))),
                fsc = Scalar[Self.dtype](atof(cols[3])),
                particle_fsc = Scalar[Self.dtype](atof(cols[4])),
                # NOTE: remove the sqrt from SSNR colums by squaring
                particle_ssnr = Scalar[Self.dtype](atof(cols[5])**2),
                reconstruction_ssnr = Scalar[Self.dtype](atof(cols[6])**2)
            ))

    def read(mut self, *, path: String) raises:
        with open(path, "r") as file:
            self.read(contents=file.read())

    def write(
        self,
        *,
        comments: List[String] = [],
        header_comment: Bool = True,
        out s: String
    ):

        # estimate an initial capacity for the string
        # doesn't have to be perfect
        s = String(capacity_bytes=len(self.records)*7*15)

        # render the comments, if any
        for comment in comments:
            s += 'C '
            s += comment
            s += '\n'

        # render the header, if needed
        if header_comment:
            if len(comments) > 0:
                s += 'C\n'
            s += 'C SHELL RESOLUTION RING_RADIUS FSC Part_FSC Part_SSNR^0.5 Rec_SSNR^0.5\n'

        # render the records
        for record in self.records:
            record.shell.write_to(s)
            s += ' '
            record.resolution.value.write_to(s)
            s += ' '
            record.ring_radius.value.write_to(s)
            s += ' '
            record.fsc.write_to(s)
            s += ' '
            record.particle_fsc.write_to(s)
            s += ' '
            # NOTE: SSNR colums are saved as sqrt
            sqrt(record.particle_ssnr).write_to(s)
            s += ' '
            sqrt(record.reconstruction_ssnr).write_to(s)
            s += '\n'


@fieldwise_init
struct ResolutionStatisticsRecord[dtype: DType](
    Copyable,
    Movable
):
    var shell: Scalar[Self.dtype]  # colum 1
    var resolution: Ang[Scalar[Self.dtype]]  # column 2
    var ring_radius: Ang[Scalar[Self.dtype]]  # column 3
    var fsc: Scalar[Self.dtype]  # column 4
    var particle_fsc: Scalar[Self.dtype]  # column 5
    var particle_ssnr: Scalar[Self.dtype]  # column 6
    var reconstruction_ssnr: Scalar[Self.dtype]  # column 7
