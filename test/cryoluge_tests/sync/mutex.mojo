
from std.time import sleep
from std.testing import assert_equal, assert_true, assert_false

# TODO: can't use this until the compiler stops crashing
# from max.algorithm import parallelize

from cryoluge.sync import Mutex
from cryoluge.test import assert_equal_float

from cryoluge_testlib import ModuleTests, TestLocation
comptime tests = ModuleTests(__functions_in_module(), lambda () -> TestLocation: TestLocation())


struct Counter(Movable):
    var count: Int

    def __init__(out self):
        self.count = 0

    def increment(mut self):
        self.count += 1

    def delayed_increment(mut self):
        var c = self.count
        sleep(0.01)
        self.count = c + 1


def test_mutex_spin_serial() raises:

    var counter = Counter()
    var mutex = Mutex()
    comptime num_tasks = 10

    for _ in range(num_tasks):
        with mutex.lock():
            counter.increment()

    assert_equal(counter.count, num_tasks)


# TODO: the closure here causes a compiler crash! =(
# def test_mutex_spin_parallel() raises:

#     var counter = Counter()
#     var mutex = Mutex()

#     def func(_i: Int) {mutex, mut counter}:
#         with mutex.lock():
#             counter.delayed_increment()

#     var num_tasks = 10
#     parallelize(func, num_tasks)

#     assert_equal(counter.count, num_tasks)


def test_mutex_wait_serial() raises:

    var counter = Counter()
    var mutex = Mutex()
    comptime num_tasks = 10

    for _ in range(num_tasks):
        with mutex.lock[wait_ms=10]():
            counter.increment()

    assert_equal(counter.count, num_tasks)


# TODO: the closure here causes a compiler crash! =(
# def test_mutex_wait_parallel() raises:

#     var counter = Counter()
#     var mutex = Mutex()

#     def func(_i: Int) {mutex, mut counter}:
#         with mutex.lock[wait_ms=10]():
#             counter.delayed_increment()

#     var num_tasks = 10
#     parallelize(func, num_tasks)
#     # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

#     assert_equal(counter.count, num_tasks)


def test_mutexed_serial() raises:

    var mutex = Mutex(Counter())
    comptime num_tasks = 10

    for _ in range(num_tasks):
        with mutex.lock() as counter:
            counter[].increment()

    assert_equal(mutex^.unwrap().count, num_tasks)


# TODO: the closure here causes a compiler crash! =(
# def test_mutexed_parallel() raises:

#     var mutex = Mutex(Counter())
#     comptime num_tasks = 10

#     def func(_i: Int) {mutex}:
#         pass
#         # with mutex.lock() as counter:
#         #     counter[].delayed_increment()

#     parallelize(func, num_tasks)
#     # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

#     assert_equal(mutex^.unwrap().count, num_tasks)
