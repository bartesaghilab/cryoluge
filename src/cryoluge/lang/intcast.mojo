
fn intcast_i32(i: Int, out i32: Int32) raises:
    var range = (0, Int(Int32.MAX))
    if i >= range[0] and i <= range[1]:
        return Int32(i)
    else:
        raise Error("Can't convert Int (", i, ") to Int32: out of range [", range[0], ", ", range[1], "]")
