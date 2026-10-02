
@fieldwise_init
struct Block(ImplicitlyCopyable, Movable, Writable, Stringable, Equatable, Keyable):
    var id: Int64
    var name: StaticString

    @staticmethod
    def unknown(id: Int64) -> Self:
        return Self(id, "?")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ")")

    def __str__(self) -> String:
        return String.write(self)

    def __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id

    comptime Key = Int64
    def key(self) -> Int64:
        return self.id
