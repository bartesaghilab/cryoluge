

from time import perf_counter_ns


comptime NS = UInt
"""nanoseconds"""


fn _ns_to_s(ns: NS, out s: Float32):
    s = Float32(ns)/Float32(1e9)


fn now(out t: NS):
    t = perf_counter_ns()


struct Timer(
    Copyable,
    Movable
):
    var name: String
    var _started: NS

    @always_inline
    @staticmethod
    fn __init__(out self, name: String):
        self.name = name
        self._started = now()

    @always_inline
    fn stop(self) -> Timed:
        var stopped = now()
        return Timed(self.name, stopped - self._started)


struct Timed(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var name: String
    var elapsed: NS

    @always_inline
    fn __init__(out self, var name: String, elapsed: NS):
        self.name = name^
        self.elapsed = elapsed

    fn elapsed_s(self, out s: Float32):
        s = _ns_to_s(self.elapsed)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Timer[", self.name, ":  elapsed=", self.elapsed_s(), " s]")

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct Benchmark(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var name: String
    var runs: Int
    var timed: Timed

    @always_inline
    @staticmethod
    fn run[
        func: fn () capturing
    ](name: String, *, warmup_runs: Int, runs: Int, out self: Self):

        # warmup
        for _ in range(warmup_runs):
            func()

        # for reals
        var timer = Timer(name)
        for _ in range(runs):
            func()
        var timed = timer.stop()

        self = Self(name, runs, timed^)

    fn runs_per_s(self, out v: Float32):
        v = Float32(self.runs)/self.timed.elapsed_s()

    fn s_per_run(self, out v: Float32):
        v = self.timed.elapsed_s()/Float32(self.runs)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Benchmark[", self.name, ":",
            "  runs=", self.runs,
            ", elapsed=", self.timed.elapsed_s(), " s",
            ", s/r=", self.s_per_run(),
            " s, r/s=", self.runs_per_s(),
            "]"
        )

    fn __str__(self) -> String:
        return String.write(self)
