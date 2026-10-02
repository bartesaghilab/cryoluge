
import os
from memory import alloc, UnsafePointer
from collections._index_normalization import normalize_index


struct MovableList[T: Movable](
    Sized,
    Movable,
    Iterable
):
    """
    A collection type for elements that are Movable, but not Copyable.
    """

    var _data: UnsafePointer[T, MutOrigin.external]
    var _capacity: Int
    var _len: Int

    comptime DEFAULT_CAPACITY = 4
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = _MovableListIter[T, iterable_origin]

    def __init__(out self, *, capacity: Int=Self.DEFAULT_CAPACITY):
        self._data = alloc[T](capacity)
        self._capacity = capacity
        self._len = 0

    def __init__(out self, *, mut take_from: Self):
        self = Self.__init__(capacity=len(take_from))
        for i in range(len(take_from)):
            (self._data + i).init_pointee_move_from(take_from._data + i)
        self._len = take_from._len
        take_from._len = 0

    def capacity(self) -> Int:
        return self._capacity

    def _expand(mut self):

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

    def append(mut self, var elem: T):

        # make more room, if needed
        if self._len >= self._capacity:
            self._expand()

        # add the new element at the end
        (self._data + self._len).init_pointee_move(elem^)
        self._len += 1

    def append_get(mut self, var elem: T) -> ref [self] T:
        self.append(elem^)
        return (self._data + self._len - 1)[]

    # TODO: insert?

    def __len__(self) -> Int:
        return self._len

    def __getitem__[I: Indexer, //](ref self, idx: I) -> ref [self] T:
        var normalized_idx = normalize_index["MovableList", assert_always=False](idx, UInt(self._len))
        return (self._data + normalized_idx)[]

    def remove[I: Indexer, //](mut self, idx: I, out item: T):
        var normalized_idx = normalize_index["MovableList", assert_always=False](idx, UInt(self._len))
        item = (self._data + normalized_idx).take_pointee()
        for i in range(normalized_idx, self._len - 1):
            (self._data + i).init_pointee_move_from(self._data + i + 1)
        self._len -= 1

    def find(self, item: T) -> Optional[Int]:
        for i in range(self._len):
            if (self._data + i) == UnsafePointer(to=item):
                return i
        return None

    def first[predicate: def (T) -> Bool](self) -> Optional[Int]:
        for i in range(self._len):
            if predicate(self[i]):
                return i
        return None

    def __iter__(ref self) -> _MovableListIter[T,origin_of(self)]:
        return _MovableListIter(self)


struct _MovableListIter[
    mut: Bool,
    //,
    T: Movable,
    origin: Origin[mut]
](
    Copyable,
    Movable,
    Iterable,
    Iterator
):
    var _list: Pointer[MovableList[T],origin]
    var _index: Int

    comptime Element = Pointer[T,origin]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    def __init__(out self, ref [origin] list: MovableList[T]):
        self._list = Pointer(to=list)
        self._index = 0

    def __iter__(ref self) -> Self:
        return self.copy()

    @inline(.always)
    def __has_next__(self) -> Bool:
        return self._index < len(self._list[])

    @inline(.always)
    def __next_ref__(mut self) -> ref [origin] T:
        ref item = self._list[][self._index]
        self._index += 1
        return item

    @inline(.always)
    def __next__(mut self: Self) -> Self.Element:
        return Pointer(to=self.__next_ref__())
