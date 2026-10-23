
from std.atomic import Atomic
from std.memory import alloc, dealloc, Layout, Allocation
from std.time import sleep


struct Mutex[T: Movable & Deinitable = NoneType](
    Movable
):
    var _alloc: Allocation[Int]
    var _item: Self.T

    comptime _unlocked = 0
    comptime _locked = 1

    def __init__(out self)
        where Self.T == NoneType:

        self = rebind_var[Self]( Mutex(None) )

    def __init__(out self, var item: Self.T):

        # init the atomic value to unlocked
        self._alloc = alloc(Layout[Int](count=1))
        self._alloc.unsafe_ptr()[] = Self._unlocked

        self._item = item^

    def __deinit__(deinit self):
        dealloc(self._alloc^)

    def unwrap(deinit self, out item: Self.T):
        dealloc(self._alloc^)
        item = self._item^

    def _unsafe_ptr(self) -> Pointer[Int,origin_of(self._alloc._alloc).unsafe_mut_cast[True]()]:
        return self._alloc.unsafe_ptr()
            .unsafe_mut_cast[True]()

    # TODO: can we relax any of the consistencies to get better optimization?
    # TODO: implement thread parking and signaling instead of doing spin,sleep waits?

    def lock[wait_ms: Int = 0](
        self,
        out guard: MutexGuard[Self.T, origin_of(self).unsafe_mut_cast[True]()]
    ):
        while True:

            # try to set the atomic to the locked state, but only if it's currently unlocked
            var expected = Self._unlocked
            var exchanged = Atomic.compare_exchange(
                ptr = self._unsafe_ptr(),
                # SAFETY: the atomic function will make sure only one caller mutates the pointer
                expected = expected,
                desired = Self._locked
            )
            if exchanged:
                # we have the lock now!
                return MutexGuard[Self.T, origin_of(self).unsafe_mut_cast[True]()](
                    _mutex = Pointer(to=self).unsafe_mut_cast[True]()
                )
                # SAFETY: only one guard can hold the lock, so the Mutex is safe to mutate
            else:
                # don't have the lock yet: try again
                if wait_ms != 0:
                    sleep(sec=Float64(wait_ms)/Float64(1000))


@fieldwise_init
struct MutexGuard[
    T: Movable & Deinitable,
    origin_mutex: Origin[mut=True]
]:
    var _mutex: Pointer[Mutex[Self.T], Self.origin_mutex]
    
    def __deinit__(deinit self):
        Atomic.store(
            self._mutex[]._unsafe_ptr(),
            # SAFETY: the atomic function will make sure only one caller mutates the pointer
            Mutex._unlocked
        )

    # NOTE: can't return a reference here, since contexts want to return by var,
    #       so return a pointer instead
    def __enter__(mut self) -> Pointer[Self.T, origin_of(self._mutex[]._item)]:
        return Pointer(to=self._mutex[]._item)

    def __exit__(self):
        # nothing to do
        pass
