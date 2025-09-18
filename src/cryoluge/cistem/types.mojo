
@fieldwise_init
struct ColumnType(ImplicitlyCopyable, Movable, Writable, EqualityComparable):
    var id: UInt8
    var name: StaticString
    var size: Optional[UInt]

    alias text = Self(1, "text", None)
    alias int = Self(2, "int", UInt(4))
    alias float = Self(3, "float", UInt(4))
    alias bool = Self(4, "bool", UInt(1))
    alias long = Self(5, "long", UInt(8))
    alias double = Self(6, "double", UInt(8))
    alias byte = Self(7, "byte", UInt(1))
    alias vary = Self(8, "vary", None)
    alias uint = Self(9, "uint", UInt(4))

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
        @parameter
        for t in Self.all:
            if t.id == id:
                return t
        raise Error(String("Unrecognized column type id: ", id)) 

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    fn __eq__(self: Self, other: Self) -> Bool:
        return self.id == other.id
