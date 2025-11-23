
@fieldwise_init
struct CoordDomain(
    ImplicitlyCopyable,
    Movable,
    EqualityComparable,
    Writable,
    Stringable
):
    var value: Int
    var name: StaticString
    
    comptime Real = CoordDomain(0, "Real")
    comptime Fourier = CoordDomain(1, "Fourier")

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __str__(self) -> String:
        return String.write(self)


fn constrain_coord_domain[msg: String, obs: CoordDomain, exp: CoordDomain]():
    constrained[
        obs == exp,
        String(msg, " expected coord domain ", exp, ", but coord domain ", obs, " was used instead")
    ]()
