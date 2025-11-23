
import os
from memory import alloc, UnsafePointer
from collections._index_normalization import normalize_index


struct MovableList[T: Movable](Sized, Movable):
    """
    A collection type for elements that are Movable, but not Copyable.
    """

    var _data: UnsafePointer[T, MutOrigin.external]
    var _capacity: Int
    var _len: Int

    comptime DEFAULT_CAPACITY = 4

    fn __init__(out self, *, capacity: Int=Self.DEFAULT_CAPACITY):
        self._data = alloc[T](capacity)
        self._capacity = capacity
        self._len = 0

    fn capacity(self) -> Int:
        return self._capacity

    fn _expand(mut self):

        # pick the new capacity        
        var capacity = self._capacity
        if capacity == 0:
            capacity = Self.DEFAULT_CAPACITY
        capacity = capacity*2

        # allocate new storage space
        var data = alloc[T](capacity)

        # move old elements over
        for i in range(self._len):
            (data + i).init_pointee_move_from(self._data + i)

        self._data = data
        self._capacity = capacity

    fn append(mut self, var elem: T):

        # make more room, if needed
        if self._len >= self._capacity:
            self._expand()

        # add the new element at the end
        (self._data + self._len).init_pointee_move(elem^)
        self._len += 1

    # TODO: insert?

    fn __len__(self) -> Int:
        return self._len

    fn __getitem__[I: Indexer, //](ref self, idx: I) -> ref [self] T:
        var normalized_idx = normalize_index["MovableList", assert_always=False](idx, UInt(self._len))
        return (self._data + normalized_idx)[]

    fn remove[I: Indexer, //](mut self, idx: I, out item: T):
        var normalized_idx = normalize_index["MovableList", assert_always=False](idx, UInt(self._len))
        item = (self._data + normalized_idx).take_pointee()
        for i in range(normalized_idx, self._len - 1):
            (self._data + i).init_pointee_move_from(self._data + i + 1)
        self._len -= 1

    fn find(self, item: T) -> Optional[Int]:
        for i in range(self._len):
            if (self._data + i) == UnsafePointer(to=item):
                return i
        return None

    fn first[predicate: fn (T) -> Bool](self) -> Optional[Int]:
        for i in range(self._len):
            if predicate(self[i]):
                return i
        return None
