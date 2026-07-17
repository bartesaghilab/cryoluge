
from testing import assert_equal

from cryoluge.image.analysis import ResolutionStatistics


comptime funcs = __functions_in_module()

comptime dtype = DType.float64


def test_read():

    var stats = ResolutionStatistics[dtype]()

    stats.read(contents = """
C this is a comment
C this is too
C
C HEADER THINGY HERE
1 2 3 4 5 6 7
       2.00000      337.92001        0.00781        0.99244        0.99967        6.02445       76.36480
       3.00000      168.96001        0.01562        0.99916        0.99996       16.92926      206.48076
    """)

    # check the records

    assert_equal(stats.records[0].shell, 1.0)
    assert_equal(stats.records[0].resolution, 2.0)
    assert_equal(stats.records[0].ring_radius, 3.0)
    assert_equal(stats.records[0].fsc, 4.0)
    assert_equal(stats.records[0].particle_fsc, 5.0)
    assert_equal(stats.records[0].particle_ssnr, 36.0)  # squared
    assert_equal(stats.records[0].reconstruction_ssnr, 49.0)  # squared

    assert_equal(stats.records[1].shell, 2.0)
    assert_equal(stats.records[1].resolution, 337.92001)
    assert_equal(stats.records[1].ring_radius, 0.00781)
    assert_equal(stats.records[1].fsc, 0.99244)
    assert_equal(stats.records[1].particle_fsc, 0.99967)
    assert_equal(stats.records[1].particle_ssnr, 6.02445**2)  # squared
    assert_equal(stats.records[1].reconstruction_ssnr, 76.36480**2)  # squared

    assert_equal(stats.records[2].shell, 3.0)
    assert_equal(stats.records[2].resolution, 168.96001)
    assert_equal(stats.records[2].ring_radius, 0.01562)
    assert_equal(stats.records[2].fsc, 0.99916)
    assert_equal(stats.records[2].particle_fsc, 0.99996)
    assert_equal(stats.records[2].particle_ssnr, 16.92926**2)  # squared
    assert_equal(stats.records[2].reconstruction_ssnr, 206.48076**2)  # squared

    assert_equal(len(stats.records), 3)
