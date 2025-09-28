
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from os import abort


struct MovableOptional[T: Movable](Movable):
    var _has: Bool
    var _v: UnsafeMaybeUninitialized[T]

    fn __init__(out self):
        self._has = False
        self._v = UnsafeMaybeUninitialized[T]()

    @implicit
    fn __init__(out self, v: NoneType):
        self = Self()

    # apparently we need to accept the magic compiler NoneType too
    @doc_private
    @implicit
    fn __init__(out self, v: NoneType._mlir_type):
        self = Self(v=NoneType(v))

    @implicit
    fn __init__(out self, var v: T):
        self._has = True
        self._v = UnsafeMaybeUninitialized[T](v^)

    fn __del__(deinit self):
        if self._has:
            self._v.assume_initialized_destroy()

    fn __moveinit__(out self, deinit other: Self):
        self._has = other._has
        self._v = UnsafeMaybeUninitialized[T]()
        if self._has:
            self._v.move_from(other._v)

    fn __is__(self, other: NoneType) -> Bool:
        return not self._has

    fn __isnot__(self, other: NoneType) -> Bool:
        return self._has

    fn value(ref self) -> ref [self._v] T:
        if not self._has:
            abort("MovableOptional was None")
        return self._v.assume_initialized()

    fn take(mut self, out v: T):
        if not self._has:
            abort("MovableOptional was None")
        self._has = False
        v = self._v.unsafe_ptr().take_pointee()

    fn or_else[func: fn () capturing -> T](mut self, out v: T):
        if self._has:
            v = self.take()
        else:
            v = func()

    fn map[
        R: Movable, //,
        func: fn(T) capturing -> R
    ](self) -> MovableOptional[R]:
        if self._has:
            return MovableOptional[R](func(self.value()))
        else:
            return MovableOptional[R]()
