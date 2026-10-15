
from std.hashlib import Hasher

from cryoluge.collections import Keyable


@fieldwise_init
struct Block(ImplicitlyCopyable, Movable, Writable, Equatable, Hashable, Keyable):
    var id: Int64
    var name: StaticString

    @staticmethod
    def unknown(id: Int64) -> Self:
        return Self(id, "?")

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ")")

    def __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id

    def __hash__(self, mut hasher: Some[Hasher]):
        self.id.__hash__(hasher)

    comptime Key = Int64
    def key(self) -> Self.Key:
        return self.id
