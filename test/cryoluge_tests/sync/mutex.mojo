
from time import sleep
from algorithm import parallelize

from testing import assert_equal, assert_true, assert_false

from cryoluge.sync import Mutex

from cryoluge_testlib import assert_equal_float


comptime funcs = __functions_in_module()


def test_mutex():

    var counter = 0
    var mutex = Mutex()

    @parameter
    fn delayed_increment(_i: Int) capturing:
        with mutex.lock():
            var c = counter
            sleep(0.1)
            counter = c + 1

    var num_tasks = 10
    parallelize[delayed_increment](num_tasks)
    # NOTE: parallelize uses a thread pool maintained by the Mojo runtime library

    _ = mutex  # TEMP: extend lifetimes to work around compiler bug

    assert_equal(counter, num_tasks)
