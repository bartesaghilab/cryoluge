
# NOTE: because `Identifiable` is already used in the stdlib =(
trait Keyable:
    alias Key: Copyable & Movable & Hashable & EqualityComparable & Writable & Stringable
    fn key(self) -> Self.Key: ...


struct KeyableSet[
    V: Keyable & Copyable & Movable
](Copyable, Movable, Sized):
    var _dict: Dict[V.Key,V]

    fn __init__(out self):
        self._dict = {}

    # TODO: can we use Iterator trait here?
    fn __init__(out self, values: List[V]) raises:
        self._dict = {}
        for value in values:
            self._add(value)

    # TODO: Iterable in fn arg pos not stable yet
    #       see: https://forum.modular.com/t/iterable-trait-as-function-argument/2284/2
    # fn __init__[I: Iterable](out self, params: I) raises:
    #     self._dict = {}
    #     for param in params:
    #         # can't express bounds on trait associated aliases, yet, so need to rebind iterator value
    #         ref p = rebind[Parameter](param)
    #         self._add(p)

    fn __init__(out self, *values: V) raises:
        self._dict = {}
        for v in values:
            self._add(v)

    fn _add(mut self, value: V) raises:
        var key = value.key()
        if key in self._dict:
            raise Error("Duplicate value in set: ", key)
        self._dict[key.copy()] = value.copy()

    fn __getitem__(self, key: V.Key) -> Optional[Pointer[V, origin=ImmutableOrigin.cast_from[__origin_of(self._dict.__getitem__(key))]]]:
        try:
            return Pointer(to=self._dict[key])
        except:
            return None

    fn __len__(self) -> Int:
        return len(self._dict)
