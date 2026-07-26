
from testing import assert_equal

from cryoluge.math import Vec, Matrix, AlignedBox, OrientedBox
from cryoluge.math.units import Deg


comptime funcs = __functions_in_module()

comptime dtype = DType.float32
comptime V1 = Vec[1,Scalar[dtype]]
comptime V2 = Vec[2,Scalar[dtype]]
comptime M1 = Matrix[1,1,dtype]
comptime M2 = Matrix[2,2,dtype]


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
    # TODO: can't compile this yet
    # var obb3 = OrientedBox(
    #     origin = V2(x=0, y=0),
    #     sizes = V2(x=1, y=2),
    #     orientation = M2(rotate=Deg[dtype](30))
    # )
    # var aabb3 = obb3.bounding_box()
    # assert_equal(aabb3.origin, V2(x=0, y=0))
    # assert_equal(aabb3.sizes, V2(x=1, y=1))  # TODO: pick right answers
