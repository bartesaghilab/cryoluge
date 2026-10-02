
from hashlib import Hasher, default_hasher
from bit import next_power_of_two
from collections.dict import _DictKeyIter


struct MovableDict[
    K: Copyable & Movable & Hashable & Equatable,
    V: Movable,
    H: Hasher = default_hasher
](Movable, Sized):
    # Piggy-back off of the stdlib Dict implementation,
    # since it's complicated enough that we don't want to copy it.
    # This struct won't be the most efficient implementation,
    # since it does a heap allocation for every insertion,
    # but it's probably good enough for now.
    var _dict: Dict[K,Self.ValuePtr,H]

    comptime initial_capacity = 8
    comptime ValuePtr = UnsafePointer[V, MutOrigin.external]

    def __init__(out self):
        self = Self(capacity=Self.initial_capacity)
    
    def __init__(out self, *, capacity: Int):

        # clamp the capacity
        var cap = capacity
        if cap < Self.initial_capacity:
            cap = Self.initial_capacity

        # coerce the capacity to a power of two
        cap = next_power_of_two(cap)

        self._dict = Dict[K,Self.ValuePtr,H](power_of_two_initial_capacity=cap)

    def __init__(out self, *, mut take_from: Self):
        self = Self.__init__(capacity=len(take_from))
        for key in take_from.key_list():
            try:
                self._dict[key.copy()] = take_from._dict.pop(key)
            except KeyError:
                pass

    def __getitem__(ref self, key: K) raises -> ref [self._dict.__getitem__(key)] V:
        var p = self._dict[key]
        return p[]

    def get[origin: Origin](ref [origin] self, key: K) -> Optional[Pointer[V, origin]]:
        try:
            ref v = self._dict[key]
            var p = v
                .unsafe_mut_cast[origin.mut]()
                .unsafe_origin_cast[origin]()
            return Pointer(to=p[])
        except KeyError:
            return None

    def __setitem__(mut self, var key: K, var val: V):

        # move the value into a pointer
        var p = alloc[V](1)
        p.init_pointee_move(val^)

        # then move the pointer into the dict
        self._dict[key^] = p

    def set_get(mut self, var key: K, var elem: V) -> ref [self] V:
        self[key.copy()] = elem^
        return self.get(key^).value()[]

    def __len__(self) -> Int:
        return len(self._dict)

    def __contains__(self, key: K) -> Bool:
        return key in self._dict

    def pop(mut self, key: K) -> MovableOptional[V]:

        # get the stored pointer, if any
        try:
            p = self._dict.pop(key)
        except KeyError:
            return None

        # take the value out of the pointer
        return p.take_pointee()

    def get_or_insert[func: def () capturing -> V](
        mut self,
        key: K
    ) -> ref [self._dict.__getitem__(key)] V:
        if not key in self:
            self[key.copy()] = func()
        return self._dict.get(key).value()[]

    def keys(self) -> _DictKeyIter[K,Self.ValuePtr,H,origin_of(self._dict)]:
        return self._dict.keys()

    def key_list(self) -> List[K]:
        """
        Returns all the keys in an owned list.
        Useful for when you want to destructively iterate over the dict.
        """
        var out = List[K](capacity=len(self._dict))
        for k in self._dict.keys():
            out.append(k.copy())
        return out^

    # NOTE: can't implement any iterators over values,
    #       since the Iterator trait requires values to be Copyable =(
