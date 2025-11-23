
# NOTE: because `Identifiable` is already used in the stdlib =(
trait Keyable:
    comptime Key: Copyable & Movable & Hashable & EqualityComparable & Writable & Stringable
    fn key(self) -> Self.Key: ...


struct KeyableSet[V: Keyable & Copyable & Movable](
    Copyable,
    Movable,
    Sized
):
    var _dict: Dict[V.Key,V]

    fn __init__(out self):
        self._dict = {}

    fn __init__(out self, *values: V) raises:
        self._dict = {}
        for v in values:
            self._add(v)

    fn __init__[I: Iterable](out self, params: I) raises:
        self._dict = {}
        for param in params:
            # can't express bounds on trait associated types, yet, so need to rebind iterator value
            ref p = rebind[V](param)
            self._add(p)

    fn _add(mut self, value: V) raises:
        var key = value.key()
        if key in self._dict:
            raise Error("Duplicate value in set: ", key)
        self._dict[key.copy()] = value.copy()

    fn __getitem__(ref self, key: V.Key) -> Optional[Pointer[V, origin_of(self._dict.__getitem__(key))]]:
        try:
            return Pointer(to=self._dict[key])
        except:
            return None

    fn __len__(self) -> Int:
        return len(self._dict)
