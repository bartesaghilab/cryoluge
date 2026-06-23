
from time import sleep
from algorithm import parallelize

from testing import assert_equal, assert_true, assert_false

from cryoluge.sync import Mutex

from cryoluge.test import assert_equal_float


comptime funcs = __functions_in_module()


struct Counter(Movable):
    var count: Int

    fn __init__(out self):
        self.count = 0

    fn delayed_increment(mut self):
        var c = self.count
        sleep(0.01)
        self.count = c + 1


def test_mutex_spin():

    var counter = Counter()
    var mutex = Mutex()

    @parameter
    fn func(_i: Int) capturing:
        with mutex.lock():
            counter.delayed_increment()

    var num_tasks = 10
    parallelize[func](num_tasks)
    # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

    _ = mutex  # TEMP: extend lifetimes to work around compiler bug

    assert_equal(counter.count, num_tasks)


def test_mutex_wait():

    var counter = Counter()
    var mutex = Mutex()

    @parameter
    fn func(_i: Int) capturing:
        with mutex.lock[wait_ms=10]():
            counter.delayed_increment()

    var num_tasks = 10
    parallelize[func](num_tasks)
    # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

    _ = mutex  # TEMP: extend lifetimes to work around compiler bug

    assert_equal(counter.count, num_tasks)


def test_mutexed():

    var counter = Counter()
    var mutex = Mutex(counter^)

    @parameter
    fn func(_i: Int) capturing:
        with mutex.lock() as counter:
            counter[].delayed_increment()

    var num_tasks = 10
    parallelize[func](num_tasks)
    # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

    assert_equal(mutex^.unwrap().count, num_tasks)
