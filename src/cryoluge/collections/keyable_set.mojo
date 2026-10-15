
from std.collections.dict import _DictValueIter
from std.hashlib import default_hasher


trait Keyable:
    comptime Key: Copyable & Deinitable & Hashable & Equatable & Writable
    def key(self) -> Self.Key: ...


struct KeyableSet[V: Keyable & Copyable & Deinitable](
    Copyable,
    Movable,
    Sized
):
    var _dict: Dict[Self.K,Self.V]

    comptime K = Self.V.Key

    def __init__(out self):
        self._dict = {}

    def __init__(out self, *values: Self.V) raises:
        self = Self()
        for v in values:
            self.add(v)

    def __init__[I: Iterable](out self, values: I) raises:

        # constraints can't quite seem to handle this yet, so use a comptime assert
        comptime iter_elem = I.IteratorType[origin_of(values)].Element
        comptime assert iter_elem == Self.V,
            t"Expects iterable with element type: {reflect[Self.V].name()}, instead got: {reflect[iter_elem].name()}"

        self = Self()
        for var val in values:
            ref v = rebind_var[Self.V](val^)
            self.add(v)

    def add(mut self, value: Self.V) raises:
        var key = value.key()
        if key in self._dict:
            raise Error(t"Duplicate value in set: {key}")
        self._dict[key.copy()] = value.copy()

    def __getitem__(ref self, key: Self.K) raises -> ref [origin_of(self._dict[key])] Self.V:
        return self._dict[key]

    def get(ref self, key: Self.K) -> Optional[Pointer[Self.V, origin_of(self._dict[key])]]:
        try:
            return Pointer(to=self._dict[key])
        except KeyError:
            return None

    def __len__(self) -> Int:
        return len(self._dict)

    def values(ref self) -> _DictValueIter[Self.K,Self.V,default_hasher,origin_of(self._dict)]:
        return self._dict.values()
