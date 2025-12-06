
from os import abort
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
    var _started: NS

    @always_inline
    @staticmethod
    fn __init__(out self):
        self._started = now()

    @always_inline
    fn stop(self) -> Timed:
        return Timed(now() - self._started)


struct Timed(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var elapsed: NS

    @always_inline
    fn __init__(out self, elapsed: NS):
        self.elapsed = elapsed

    fn elapsed_s(self, out s: Float32):
        s = _ns_to_s(self.elapsed)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Timer[elapsed=", self.elapsed_s(), " s]")

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct Benchmark(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var runs: Int
    var timed: Timed

    @always_inline
    @staticmethod
    fn run[
        func: fn () capturing
    ](*, warmup_runs: Int, runs: Int, out self: Self):

        # warmup
        for _ in range(warmup_runs):
            func()

        # for reals
        var timer = Timer()
        for _ in range(runs):
            func()
        var timed = timer.stop()

        self = Self(runs, timed^)

    fn runs_per_s(self, out v: Float32):
        v = Float32(self.runs)/self.timed.elapsed_s()

    fn s_per_run(self, out v: Float32):
        v = self.timed.elapsed_s()/Float32(self.runs)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Benchmark["
            "  runs=", self.runs,
            ", elapsed=", self.timed.elapsed_s(), " s",
            ", s/r=", self.s_per_run(),
            " s, r/s=", self.runs_per_s(),
            "]"
        )

    fn __str__(self) -> String:
        return String.write(self)


struct Profiler(
    Writable,
    Stringable
):
    var _counters: List[ProfilerCounter]
    var _lookup: Dict[String,Int]

    fn __init__(out self):
        self._counters = []
        self._lookup = {}

    fn counter(mut self, name: String) -> ref [self._counters] ProfilerCounter:
        var i = self._lookup.get(name)
        if i is None:
            print("add counter", name)  # TEMP
            i = len(self._counters)
            self._counters.append(ProfilerCounter(name))
            self._lookup[name] = i.value()
        return self._counters[i.value()]

    fn start(mut self, name: String):
        self.counter(name).start()

    fn stop(mut self, name: String):
        self.counter(name).stop()
        
    fn switch(mut self, stop: String, start: String):
        self.stop(stop)
        self.start(start)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Profiler[")
        var count = 0
        for counter in self._counters:
            if count > 0:
                writer.write("  ")
            count += 1
            writer.write(counter.name, "=", counter.elapsed_s(), " s")
        writer.write("]")

    fn __str__(self) -> String:
        return String.write(self)


struct ProfilerCounter(Copyable, Movable):
    var name: String
    var _elapsed: NS
    var _start: Optional[NS]

    fn __init__(out self, name: String):
        self.name = name
        self._elapsed = 0
        self._start = None

    fn start(mut self):
        if self._start is None:
            self._start = now()
        else:
            abort(String("Counter ", self.name, " already started"))

    fn stop(mut self, *, verbose: Bool = False):
        if self._start is None:
            abort(String("Counter ", self.name, " not started"))
        self._elapsed += now() - self._start.value()
        if verbose:
            print("ProfilerCounter[", self.name, ": elapsed=", self.elapsed_s(), " ")
        self._start = None
    
    fn elapsed_s(self, out s: Float32):
        s = _ns_to_s(self._elapsed)
