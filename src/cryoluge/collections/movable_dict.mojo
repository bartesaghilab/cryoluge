
from hashlib import Hasher, default_hasher
from bit import next_power_of_two


struct MovableDict[
    K: Copyable & Movable & Hashable & EqualityComparable,
    V: Movable,
    H: Hasher = default_hasher
](Movable, Sized):
    # Piggy-back off of the stdlib Dict implementation,
    # since it's complicated enough that we don't want to copy it.
    # This struct won't be the most efficient implementation,
    # since it does a heap allocation for every insertion,
    # but it's probably good enough for now.
    var _dict: Dict[K,UnsafePointer[V],H]

    alias initial_capacity = 8

    fn __init__(out self):
        self = Self(capacity=Self.initial_capacity)
    
    fn __init__(out self, *, capacity: UInt):

        # clamp the capacity
        var cap = capacity
        if cap < Self.initial_capacity:
            cap = Self.initial_capacity

        # coerce the capacity to a power of two
        cap = next_power_of_two(cap)

        self._dict = Dict[K,UnsafePointer[V],H](power_of_two_initial_capacity=cap)

    fn __getitem__(self, key: K) raises -> ref [self._dict.__getitem__(key)] V:
        var p = self._dict[key]
        return p[]

    fn __setitem__(mut self, var key: K, var val: V):

        # move the value into a pointer
        var p = UnsafePointer[V]().alloc(1)
        p.init_pointee_move(val^)

        # then move the pointer into the dict
        self._dict[key^] = p

    fn __len__(self) -> Int:
        return len(self._dict)

    fn __contains__(self, key: K) -> Bool:
        return key in self._dict

    fn pop(mut self, key: K) -> MovableOptional[V]:

        # get the stored pointer, if any
        try:
            p = self._dict.pop(key)
        except KeyError:
            return None

        # take the value out of the pointer
        return p.take_pointee()

    fn get_or_insert[func: fn () capturing -> V](
        mut self,
        key: K
    ) -> ref [self._dict.__getitem__(key)] V:
        if not key in self:
            self[key.copy()] = func()
        return self._dict.get(key).value()[]
