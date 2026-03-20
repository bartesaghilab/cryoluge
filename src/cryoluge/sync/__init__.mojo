
from sys.ffi import external_call
from python import Python
from python._cpython import GILReleased

from .mutex import Mutex


fn thread_id() -> UInt64:
    return external_call[
        "gettid",
        UInt64
    ]()


struct DropGIL:
    var _released: GILReleased

    fn __init__(out self):
        self._released = GILReleased(Python())

    fn __enter__(mut self):
        self._released.__enter__()
    
    fn __exit__(mut self):
        self._released.__exit__()
