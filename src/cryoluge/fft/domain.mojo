
@fieldwise_init
struct CoordDomain(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable,
    Stringable
):
    var value: Int
    var name: StaticString
    
    comptime Real = CoordDomain(0, "Real")
    comptime Fourier = CoordDomain(1, "Fourier")

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    def __str__(self) -> String:
        return String.write(self)


def constrain_coord_domain[msg: String, obs: CoordDomain, exp: CoordDomain]():
    constrained[
        obs == exp,
        String(msg, " expected coord domain ", exp, ", but coord domain ", obs, " was used instead")
    ]()
