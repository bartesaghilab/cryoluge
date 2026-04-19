
from testing import assert_equal

from cryoluge.sync import Cpus, VirtualCoreSet


comptime funcs = __functions_in_module()


def test_get_all():
    var cpus = Cpus.get_all()
    print(cpus)
    # just don't crash


def test_get_allowed():
    var cpus = Cpus.get_allowed()
    print(cpus)
    # just don't crash


def test_set_get_affinity():

    try:
        # set to just a couple cores
        VirtualCoreSet([0, 1]).set_affinity()

        var affinity = VirtualCoreSet.get_affinity().to_virtual_core_ids()
        assert_equal(affinity, [0, 1])

    finally:
        # put it back to normal
        VirtualCoreSet(Cpus.get_allowed().virtual_cores()).set_affinity()


def test_partition_vcores():

    var cpus = Cpus.get_all()
    var all_vcores = cpus.virtual_cores()
    var even_vcores = cpus.virtual_cores(even=True, odd=False)
    var odd_vcores = cpus.virtual_cores(even=False, odd=True)

    assert_equal(len(even_vcores), len(all_vcores)//2)
    assert_equal(len(odd_vcores), len(all_vcores)//2)

    assert_equal(even_vcores[0].virtual_id, 0)
    assert_equal(odd_vcores[0].virtual_id, 1)
