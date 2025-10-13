
fn is_even(n: Int) -> Bool:
    return (n & 1) == 0


fn is_odd(n: Int) -> Bool:
    return not is_even(n)
