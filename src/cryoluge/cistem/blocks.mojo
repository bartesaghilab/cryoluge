
@fieldwise_init
struct Block(ImplicitlyCopyable, Movable, Writable, Stringable, EqualityComparable, Keyable):
    var id: Int64
    var name: StaticString

    @staticmethod
    fn unknown(id: Int64) -> Self:
        return Self(id, "?")

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ")")

    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id

    comptime Key = Int64
    fn key(self) -> Int64:
        return self.id
