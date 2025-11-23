
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from os import abort


struct MovableOptional[T: Movable](Movable):
    var _has: Bool
    var _v: UnsafeMaybeUninitialized[T]

    comptime default_abort_msg = "MovableOptional was None"

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

    fn has_or_abort(self, msg: String = Self.default_abort_msg):
        if not self._has:
            abort(msg)

    fn value(ref self, *, msg: String = Self.default_abort_msg) -> ref [self._v._array] T:
        self.has_or_abort(msg)
        return self._v.assume_initialized()

    fn take(mut self, out v: T, *, msg: String = Self.default_abort_msg):
        self.has_or_abort(msg)
        self._has = False
        v = self._v.unsafe_ptr().take_pointee()

    fn unwrap(var self, out v: T, *, msg: String = Self.default_abort_msg):
        v = self.take(msg=msg)

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
