
from sys.ffi import external_call
from python import Python
from python._cpython import GILReleased, GILAcquired

from .mutex import Mutex
from .affinity import Cpus, Cpu, VirtualCore, VirtualCoreSet


def process_id() -> UInt64:
    return external_call[
        "getpid",
        UInt64
    ]()


def thread_id() -> UInt64:
    return external_call[
        "gettid",
        UInt64
    ]()


struct DropGIL:
    var _released: GILReleased

    def __init__(out self):
        self._released = GILReleased(Python())

    def __enter__(mut self):
        self._released.__enter__()
    
    def __exit__(mut self):
        self._released.__exit__()


struct HoldGIL:
    var _acquired: GILAcquired

    def __init__(out self):
        self._acquired = GILAcquired(Python())

    def __enter__(mut self):
        self._acquired.__enter__()
    
    def __exit__(mut self):
        self._acquired.__exit__()
