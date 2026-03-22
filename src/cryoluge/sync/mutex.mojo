
from os.atomic import Atomic, Consistency
from memory import alloc, UnsafePointer
from time import sleep


struct Mutex[
    T: AnyType & Movable = NoneType
](
    Movable
):
    var _ptr: Self._Ptr
    var _item: T

    alias _Val = Scalar[DType.int]
    alias unlocked = Self._Val(0)
    alias locked = Self._Val(1)

    alias _Ptr = UnsafePointer[Self._Val,MutOrigin.external]


    fn __init__(out self: Mutex[NoneType]):
        self = Mutex[NoneType](None)

    fn __init__(out self, var item: T):

        # init the atomic value to unlocked
        self._ptr = alloc[Self._Val](1)
        self._ptr[] = Self.unlocked

        self._item = item^

    # TODO: can we relax any of the consistencies to get better optimization?
    # TODO: implement thread parking and signaling instead of doing spin,sleep waits?

    fn lock[wait_ms: Int = 0](
        mut self,
        out guard: MutexGuard[T,origin_of(self._item)]
    ):
        while True:

            # try to set the atomic to the locked state, but only if it's currently unlocked
            var expected = Self.unlocked
            var exchanged = Atomic.compare_exchange[
                failure_ordering = Consistency.SEQUENTIAL,
                success_ordering = Consistency.SEQUENTIAL
            ](
                ptr = self._ptr,
                expected = expected,
                desired = Self.locked
            )
            if exchanged:
                # we have the lock now!
                return MutexGuard[T](self._ptr, self._item)
            else:
                # don't have the lock yet: try again
                if wait_ms != 0:
                    sleep(sec=Float64(wait_ms)/Float64(1000))

    fn __del__(deinit self):
        self._ptr.free()

    fn unwrap(deinit self, out item: T):
        item = self._item^


struct MutexGuard[
    T: AnyType & Movable,
    item_origin: Origin[mut=True]
]:
    var _ptr: Mutex._Ptr
    var _item: Pointer[T,item_origin]

    fn __init__(
        out self,
        ptr: Mutex._Ptr,
        ref [item_origin] item: T
    ):
        self._ptr = ptr
        self._item = Pointer(to=item)

    fn __del__(deinit self):
        # set the atomic to unlocked
        Atomic.store[
            ordering = Consistency.SEQUENTIAL
        ](
            ptr = self._ptr,
            value = Mutex.unlocked
        )

    # NOTE: this doesn't work, possibly due to a compiler bug?
    #       the returned reference is always immutable
    #       despite both origins being explicity mutable =(
    # fn __enter__(mut self) -> ref [item_origin] T:
    #     return self._item[]

    fn __enter__(self) -> Pointer[T,item_origin]:
        return self._item

    fn __exit__(self):
        # nothing to do
        pass
