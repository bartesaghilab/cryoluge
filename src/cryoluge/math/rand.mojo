
from random import Random
from os import abort
from sys import size_of
from utils.numerics import FPUtils, isfinite

# TEMP
import random

from cryoluge.math import Dimension, Vec


comptime default_seed = 12345


struct Rand:
    """
    A RNG with locally-stored state.
    Useful for multi-threaded programs that don't want each thread
    to share the global RNG in Mojo's stdlib.

    WARNING: This is not a CSPRNG, and therefore is not suitable when you
             want (nearly) unpredictable randomness.
             Meaning, don't use this for cryptography!
    """

    var _rand: Random[Self._num_rounds]
    var _cache: SIMD[DType.uint32, Self._cache_length]
    var _cache_index: Int

    comptime _num_rounds = 10
    comptime _cache_length = 4

    fn __init__(
        out self,
        *,
        seed: Int = default_seed
    ):
        self._rand = Random[Self._num_rounds](seed=UInt64(seed))
        self._cache = SIMD[DType.uint32, Self._cache_length](0)
        self._cache_index = Self._cache_length

    fn uint32(mut self, out v: UInt32):
        if self._cache_index >= Self._cache_length:
            self._cache = self._rand.step()
            self._cache_index = 0
        v = self._cache[self._cache_index]
        self._cache_index += 1

    fn uint64(mut self, out v: UInt64):
        var high = UInt64(self.uint32())
        var low = UInt64(self.uint32())
        v = (high << 32) | low

    fn float64(mut self, out v: Float64):
        """
        Returns a random float64 in [0,1).
        """

        comptime f64_mantissa_bits = FPUtils[DType.float64].mantissa_width()
        comptime f64_sign_bits = 1
        comptime f64_bits = f64_mantissa_bits + f64_sign_bits
        # TEMP: can't run this at comptime for some reason: "not implemented"
        #comptime normalization = 1.0/Float64(1 << f64_bits)
        var normalization = 1.0/Float64(1 << f64_bits)

        # get enough bits of randomness to cover the mantissa and the sign (but not the exponent)
        var value = self.uint64() >> (size_of[Float64]()*8 - f64_bits)

        # normalize the value to get a float in [0,1)
        v = Float64(value)*normalization

    @always_inline
    fn float64(mut self, *, min: Float64, max: Float64, out v: Float64):
        """
        Returns a random float64 in [min,max).
        Min and max must be finite.
        """
        debug_assert(isfinite(min), "Min must be finite")
        debug_assert(isfinite(max), "Max must be finite")
        v = min + self.float64()*(max - min)

    fn float32(mut self, out v: Float32):
        """
        Returns a random float32 in [0,1).
        """

        comptime f32_mantissa_bits = FPUtils[DType.float32].mantissa_width()
        comptime f32_sign_bits = 1
        comptime f32_bits = f32_mantissa_bits + f32_sign_bits
        comptime normalization = 1.0/Float32(1 << f32_bits)

        # get enough bits of randomness to cover the mantissa and the sign (but not the exponent)
        var value = self.uint32() >> (size_of[Float32]()*8 - f32_bits)

        # normalize the value to get a float in [0,1)
        v = Float32(value)*normalization

    @always_inline
    fn float32(mut self, *, min: Float32, max: Float32, out v: Float32):
        """
        Returns a random float32 in [min,max).
        Min and max must be finite.
        """
        debug_assert(isfinite(min), "Min must be finite")
        debug_assert(isfinite(max), "Max must be finite")
        v = min + self.float32()*(max - min)

    fn scalar[dtype: DType](mut self, out v: Scalar[dtype]):
        """
        Returns a random dtype (either float32 or float64) in [-,1).
        """
        @parameter
        if dtype == DType.float32:
            v = rebind[Scalar[dtype]](self.float32())
        elif dtype == DType.float64:
            v = rebind[Scalar[dtype]](self.float64())
        else:
            constrained[False, String("Rand not supported yet for dtype: ", dtype)]()
            v = 0
            abort()

    @always_inline
    fn scalar[dtype: DType](mut self, *, min: Scalar[dtype], max: Scalar[dtype], out v: Scalar[dtype]):
        """
        Returns a random dtype (either float32 or float64) in [min,max).
        Min and max must be finite.
        """
        debug_assert(isfinite(min), "Min must be finite")
        debug_assert(isfinite(max), "Max must be finite")
        v = min + self.scalar[dtype]()*(max - min)


struct VecRand[
    dtype: DType,
    dim: Dimension
]:
    """
    A utility to easily generate random numbers in multi-dimensional ranges.

    WARNING: This is not a CSPRNG, and therefore is not suitable when you
            want (nearly) unpredictable randomness.
            Meaning, don't use this for cryptography!
    """

    var min: Self.Vec
    var max: Self.Vec
    var _rand: Rand

    comptime Vec = Vec[Scalar[dtype],dim]

    fn __init__(
        out self,
        *,
        min: Self.Vec,
        max: Self.Vec,
        seed: Int = default_seed
    ):
        self.min = min.copy()
        self.max = max.copy()
        self._rand = Rand(seed=seed)

    @always_inline
    fn next(mut self, d: Int, out v: Scalar[dtype]):
        v = self._rand.scalar[dtype](min=self.min[d], max=self.max[d])
