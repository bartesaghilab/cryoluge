
from sys.info import is_big_endian, is_little_endian
from os import abort


struct Endian(
    Copyable,
    EqualityComparable
):
    var value: Int

    alias Big=Endian(0)
    alias Little=Endian(1)

    @staticmethod
    fn native() -> Self:
        @parameter
        if is_big_endian():
            return Self.Big
        elif is_little_endian():
            return Self.Little
        else:
            constrained[False, "Target architecture is neither big nor little endian"]()
            return abort[Self]()

    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, rhs: Endian) -> Bool:
        return self.value == rhs.value

    fn __ne__(self, rhs: Endian) -> Bool:
        return self.value != rhs.value
