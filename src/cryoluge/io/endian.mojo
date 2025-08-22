

struct Endian(
    Copyable,
    EqualityComparable
):
    var value: Int

    alias Big=Endian(0)
    alias Little=Endian(1)

    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, rhs: Endian) -> Bool:
        return self.value == rhs.value

    fn __ne__(self, rhs: Endian) -> Bool:
        return self.value != rhs.value
