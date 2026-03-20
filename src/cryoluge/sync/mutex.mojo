
from os.atomic import Atomic, Consistency
from memory import alloc, UnsafePointer
from time import sleep


struct Mutex(
    Movable
):
    var ptr: UnsafePointer[Self.Val,MutOrigin.external]

    alias Val = Scalar[DType.int]
    alias unlocked = Self.Val(0)
    alias locked = Self.Val(1)


    fn __init__(out self):
        # init the atomic value to unlocked
        self.ptr = alloc[Self.Val](1)
        self.ptr[] = Self.unlocked

    # TODO: can we relax any of the consistencies to get better optimization?
    # TODO: implement thread parking and signaling instead of doing spin,sleep waits?

    fn lock[wait_ms: Int = 0](
        mut self,
        out guard: MutexGuard[origin_of(self)]
    ):
        while True:

            # try to set the atomic to the locked state, but only if it's currently unlocked
            var expected = Self.unlocked
            var exchanged = Atomic.compare_exchange[
                failure_ordering = Consistency.SEQUENTIAL,
                success_ordering = Consistency.SEQUENTIAL
            ](
                ptr = self.ptr,
                expected = expected,
                desired = Self.locked
            )
            if exchanged:
                # we have the lock now!
                return MutexGuard(self)
            else:
                # don't have the lock yet: try again
                if wait_ms != 0:
                    sleep(sec=Float64(wait_ms)/Float64(1000))

    fn __del__(deinit self):
        self.ptr.free()


struct MutexGuard[
    origin: Origin[mut=True]
]:
    var mutex: Pointer[Mutex,origin]

    fn __init__(out self, ref [origin] mutex: Mutex):
        self.mutex = Pointer(to=mutex)

    fn __del__(deinit self):
        # set the atomic to unlocked
        Atomic.store[
            ordering = Consistency.SEQUENTIAL
        ](
            ptr = self.mutex[].ptr,
            value = Mutex.unlocked
        )

    fn __enter__(self):
        # nothing to do
        pass

    fn __exit__(self):
        # nothing to do
        pass
