

@fieldwise_init
struct ColumnType(Copyable, Movable, Writable, EqualityComparable):
    var id: UInt8
    var name: String
    var size: Optional[Int]

    alias text = Self(1, "text", None)
    alias int = Self(2, "int", 4)
    alias float = Self(3, "float", 4)
    alias bool = Self(4, "bool", 1)
    alias long = Self(5, "long", 8)
    alias double = Self(6, "double", 8)
    alias byte = Self(7, "byte", 1)
    alias vary = Self(8, "vary", None)
    alias uint = Self(9, "uint", 4)

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
        var i = Int(id) - 1
        if i < 0 or i >= len(Self.all):
            raise Error(String("Unrecognized column type id: ", id)) 
        return Self.all[i]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, "(", self.id, ")")

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id
