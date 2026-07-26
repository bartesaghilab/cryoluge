
from math import sqrt
from testing import assert_equal

from cryoluge.math import Vec, Matrix, AlignedBox, OrientedBox
from cryoluge.math.units import Deg


comptime funcs = __functions_in_module()

comptime dtype = DType.float32
comptime V1 = Vec[1,Scalar[dtype]]
comptime V2 = Vec[2,Scalar[dtype]]
comptime V3 = Vec[3,Scalar[dtype]]
comptime M1 = Matrix[1,1,dtype]
comptime M2 = Matrix[2,2,dtype]
comptime M3 = Matrix[3,3,dtype]


def test_unit_corners():

    # 1D
    var corners1 = AlignedBox[1,dtype].unit_corners()
    assert_equal(corners1[0], V1(x=0))
    assert_equal(corners1[1], V1(x=1))
    assert_equal(len(corners1), 2)

    # 2D
    var corners2 = AlignedBox[2,dtype].unit_corners()
    assert_equal(corners2[0], V2(x=0, y=0))
    assert_equal(corners2[1], V2(x=1, y=0))
    assert_equal(corners2[2], V2(x=0, y=1))
    assert_equal(corners2[3], V2(x=1, y=1))
    assert_equal(len(corners2), 4)

    # 3D
    var corners3 = AlignedBox[3,dtype].unit_corners()
    assert_equal(corners3[0], V3(x=0, y=0, z=0))
    assert_equal(corners3[1], V3(x=1, y=0, z=0))
    assert_equal(corners3[2], V3(x=0, y=1, z=0))
    assert_equal(corners3[3], V3(x=1, y=1, z=0))
    assert_equal(corners3[4], V3(x=0, y=0, z=1))
    assert_equal(corners3[5], V3(x=1, y=0, z=1))
    assert_equal(corners3[6], V3(x=0, y=1, z=1))
    assert_equal(corners3[7], V3(x=1, y=1, z=1))
    assert_equal(len(corners3), 8)


def test_oriented_bound():

    # 1D is kind of trivial, but whatever
    var obb1 = OrientedBox(
        origin = V1(x=0),
        sizes = V1(x=1),
        orientation = M1.identity()
    )
    var aabb1 = obb1.bounding_box()
    assert_equal(aabb1.origin, V1(x=0))
    assert_equal(aabb1.sizes, V1(x=1))

    # test 2D, identity
    var obb2 = OrientedBox(
        origin = V2(x=0, y=0),
        sizes = V2(x=1, y=1),
        orientation = M2.identity()
    )
    var aabb2 = obb2.bounding_box()
    assert_equal(aabb2.origin, V2(x=0, y=0))
    assert_equal(aabb2.sizes, V2(x=1, y=1))

    # test 2D, non-trivial rotation
    var obb3 = OrientedBox(
        origin = V2(x=0, y=0),
        sizes = V2(x=1, y=2),
        orientation = M2(rotate=Deg[dtype](30))
    )
    var aabb3 = obb3.bounding_box()
    var r3o2 = sqrt(Scalar[dtype](3))/2
    var h = Scalar[dtype](0.5)
    assert_equal(aabb3.origin, V2(x=-h, y=0))
    assert_equal(aabb3.sizes, V2(x=h+r3o2, y=h+r3o2))
