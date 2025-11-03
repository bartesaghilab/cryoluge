
from hashlib import Hasher, default_hasher
from bit import next_power_of_two
from collections.dict import _DictKeyIter


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
    
    fn __init__(out self, *, capacity: Int):

        # clamp the capacity
        var cap = capacity
        if cap < Self.initial_capacity:
            cap = Self.initial_capacity

        # coerce the capacity to a power of two
        cap = next_power_of_two(cap)

        self._dict = Dict[K,UnsafePointer[V],H](power_of_two_initial_capacity=cap)

    fn __getitem__(ref self, key: K) raises -> ref [self._dict.__getitem__(key)] V:
        var p = self._dict[key]
        return p[]

    fn get(self, key: K) -> Optional[Pointer[V, __origin_of(self[key])]]:
        try:
            ref v = self._dict[key]
            var p = v.origin_cast[target_mut=False, target_origin=__origin_of(self[key])]()
            return Pointer(to=p[])
        except KeyError:
            return None

    fn get_mut(mut self, key: K) -> Optional[Pointer[V, __origin_of(self[key])]]:
        try:
            ref v = self._dict[key]
            var p = v.origin_cast[target_mut=True, target_origin=__origin_of(self[key])]()
            return Pointer(to=p[])
        except KeyError:
            return None

    # TODO: when pointer v2 in the stdlib ships (next release?),
    #       try to unify the two above get() impls by using parametric mutability
    #       Can't seem to find a way to do it now =(

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

    fn keys(self) -> _DictKeyIter[K,UnsafePointer[V],H,__origin_of(self._dict)]:
        return self._dict.keys()

    fn key_list(self) -> List[K]:
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
