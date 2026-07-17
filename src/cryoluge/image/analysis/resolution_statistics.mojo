
from cryoluge.math.units import Ang


# NOTE: resolution files are small enough that they can be
#       entirely buffered with no real performance considerations


struct ResolutionStatistics[dtype: DType]:
    var records: List[ResolutionStatisticsRecord[dtype]]

    fn __init__(out self):
        self.records = []

    fn read(mut self, *, contents: String) raises:

        self.records.clear()

        # read the file line-by-line
        var linei = 0
        for line in contents.splitlines():
            linei += 1

            # skip blank lines
            if len(line) <= 0:
                continue

            # skip comment lines
            # NOTE: String.__getitem__() splits on byte boundaries
            #       (ie, is only valid for ASCII-encoded strings),
            #       but we're checking for ASCII 'C' at the first position, so it's safe
            if line[0] == "C":
                continue

            # parse the columns
            var cols = [w for w in line.split(" ") if len(w) > 0]

            # skip empty lines
            if len(cols) <= 0:
                continue
                
            if len(cols) < 7:
                raise Error("Too few columns (", len(cols), ") on line ", linei)

            self.records.append(ResolutionStatisticsRecord(
                shell = Scalar[dtype](atof(cols[0])),
                resolution = Ang[dtype](atof(cols[1])),
                ring_radius = Ang[dtype](atof(cols[2])),
                fsc = Scalar[dtype](atof(cols[3])),
                particle_fsc = Scalar[dtype](atof(cols[4])),
                # NOTE: remove the sqrt from SSNR colums by squaring
                particle_ssnr = Scalar[dtype](atof(cols[5])**2),
                reconstruction_ssnr = Scalar[dtype](atof(cols[6])**2)
            ))

    fn read(mut self, *, path: String) raises:
        with open(path, "r") as file:
            self.read(contents=file.read())

    fn write(
        self,
        *,
        comments: List[String] = [],
        header_comment: Bool = True,
        out s: String
    ):

        # estimate an initial capacity for the string
        # doesn't have to be perfect
        s = String(capacity=len(self.records)*7*15)

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
    var shell: Scalar[dtype]  # colum 1
    var resolution: Ang[dtype]  # column 2
    var ring_radius: Ang[dtype]  # column 3
    var fsc: Scalar[dtype]  # column 4
    var particle_fsc: Scalar[dtype]  # column 5
    var particle_ssnr: Scalar[dtype]  # column 6
    var reconstruction_ssnr: Scalar[dtype]  # column 7
