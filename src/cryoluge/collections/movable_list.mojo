
struct MovableList[T: Movable]:
    """
    A collection type for elements that are Movable, but not Copyable.
    """

    var _data: UnsafePointer[T]
    var _capacity: Int
    var _len: Int

    alias DEFAULT_CAPACITY = 4

    fn __init__(out self, *, capacity: Int=Self.DEFAULT_CAPACITY):
        self._data = UnsafePointer[T]().alloc(capacity)
        self._capacity = capacity
        self._len = 0

    fn _expand(mut self):

        # pick the new capacity        
        var capacity = self._capacity
        if capacity == 0:
            capacity = Self.DEFAULT_CAPACITY
        capacity = capacity*2

        # allocate new storage space
        var data = UnsafePointer[T]().alloc(capacity)

        # move old elements over
        for i in range(self._len):
            (data + i).init_pointee_move_from(self._data + i)

        self._data = data
        self._capacity = capacity

    fn add(mut self, var elem: T):

        # make more room, if needed
        if self._len >= self._capacity:
            self._expand()

        # add the new element at the end
        (self._data + self._len).init_pointee_move(elem^)
        self._len += 1
    
    # fn remove(mut self, elem: T) -> T:
    #     # TODO: implement me!
    #     pass
    