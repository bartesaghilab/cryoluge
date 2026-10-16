
@fieldwise_init
struct CoordDomain(
    ImplicitlyCopyable,
    Movable,
    Equatable,
    Writable,
    TrivialRegisterPassable
):
    var value: Int
    var name: StaticString
    
    comptime Real = CoordDomain(0, "Real")
    comptime Fourier = CoordDomain(1, "Fourier")

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)
