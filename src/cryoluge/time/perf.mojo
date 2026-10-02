
from os import abort
from time import perf_counter_ns

from cryoluge.collections import MovableList


comptime NS = UInt
"""nanoseconds"""


def _ns_to_s(ns: NS, out s: Float32):
    s = Float32(ns)/Float32(1e9)


@inline(.always)
def now(out t: NS):
    t = perf_counter_ns()


struct Timer(
    Copyable,
    Movable
):
    var _started: NS

    @inline(.always)
    @staticmethod
    def __init__(out self):
        self._started = now()

    @inline(.always)
    def stop(self) -> Timed:
        return Timed(now() - self._started)


struct Timed(
    Copyable,
    Movable,
    Writable,
    Stringable
):
    var elapsed: NS

    @inline(.always)
    def __init__(out self, elapsed: NS):
        self.elapsed = elapsed

    def elapsed_s(self, out s: Float32):
        s = _ns_to_s(self.elapsed)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Timer[elapsed=", self.elapsed_s(), " s]")

    def __str__(self) -> String:
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

    @inline(.always)
    @staticmethod
    def run[
        func: def () capturing
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

    def runs_per_s(self, out v: Float32):
        v = Float32(self.runs)/self.timed.elapsed_s()

    def s_per_run(self, out v: Float32):
        v = self.timed.elapsed_s()/Float32(self.runs)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Benchmark["
            "runs=", self.runs,
            ", elapsed=", self.timed.elapsed_s(), " s",
            ", s/r=", self.s_per_run(),
            " s, r/s=", self.runs_per_s(),
            "]"
        )

    def __str__(self) -> String:
        return String.write(self)


struct Profiler[*, enabled: Bool = True](
    Movable,
    Writable,
    Stringable
):
    var _counters: MovableList[ProfilerCounter[enabled=enabled]]
    var _lookup: Dict[String,Int]
    var unit: StaticString

    def __init__(out self, *, unit: StaticString = 's'):
        self._counters = MovableList[ProfilerCounter[enabled=enabled]](capacity=64)
        # NOTE: use a somewhat large initial capacity for the list,
        #       otherwise, creating new counters can invalidate outstanding counter references
        #       when the list resizes
        #       it's kind of surprising the compiler even allows this ... maybe the v1.0 compiler doesn't
        #       the same problem happened on std List, so it's not just a flaw in my MovableList
        # TODO: does this issue still happen in 1.0? should I report a footgun?
        self._lookup = {}
        self.unit = unit

    def counter(mut self, name: String) -> ref [self._counters[0]] ProfilerCounter[enabled=enabled]:
        var i = self._lookup.get(name)
        if i is None:
            i = len(self._counters)
            self._counters.append(ProfilerCounter[enabled=enabled](name, self.unit))
            self._lookup[name] = i.value()
        return self._counters[i.value()]

    def start(mut self, name: String):
        @parameter
        if enabled:
            self.counter(name).start()

    def stop[*, verbose: Bool = False](mut self, name: String):
        @parameter
        if enabled:
            self.counter(name).stop[verbose=verbose]()
        
    def switch[*, verbose: Bool = False](mut self, stop: String, start: String):
        @parameter
        if enabled:
            self.stop[verbose=verbose](stop)
            self.start(start)

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        if enabled:
            writer.write("Profiler[")
            for i in range(len(self._counters)):
                if i > 0:
                    writer.write("  ")
                ref counter = self._counters[i]
                writer.write(counter.name, "=", _render_elapsed(counter.elapsed_s(), self.unit))
            writer.write("]")

    def __str__(self) -> String:
        return String.write(self)
    
    def print(self):
        @parameter
        if enabled:
            print(self)


struct ProfilerCounter[*, enabled: Bool = True](
    Movable,
    Writable,
    Stringable
):
    var name: String
    var unit: StaticString
    var _elapsed: NS
    var _start: Optional[NS]

    def __init__(out self, name: String, unit: StaticString):
        self.name = name
        self.unit = unit
        self._elapsed = 0
        self._start = None

    @inline(.always)
    def start(mut self):
        @parameter
        if enabled:
            self._start = now()

    @inline(.always)
    def stop[*, verbose: Bool = False](mut self):
        @parameter
        if enabled:
            var elapsed = now() - self._start.value()
            self._start = None
            self._elapsed += elapsed
            @parameter
            if verbose:
                print("ProfilerCounter[", self.name, ":",
                    "  elapsed=", _render_elapsed(_ns_to_s(elapsed), self.unit),
                    "  total=", _render_elapsed(self.elapsed_s(), self.unit),
                "]", sep="")
    
    def elapsed_s(self, out s: Float32):
        s = _ns_to_s(self._elapsed)

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        if enabled:
            writer.write("ProfilerCounter[",
                self.name, "=", _render_elapsed(self.elapsed_s(), self.unit),
            "]")

    def __str__(self) -> String:
        return String.write(self)


def _render_elapsed(elapsed_s: Float32, unit: StaticString) -> String:
    if unit == 'us':
        return String(elapsed_s*1000*1000, " us")
    elif unit == 'ms':
        return String(elapsed_s*1000, " ms")
    else:
        return String(elapsed_s, " s")
