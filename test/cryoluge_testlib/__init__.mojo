
from std.collections import Deque
from std.io import FileHandle
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite as StdTestSuite
from std.testing.suite import TestSuiteReport
from std.reflection import call_location, SourceLocation
from std.pathlib import Path


def file_handle(tempfile: NamedTemporaryFile) -> ref [tempfile._file_handle] FileHandle:
    return tempfile._file_handle
    # NOTE: _file_handle is internal, and therefore probably unstable?


@fieldwise_init
struct ModuleTests[
    Locator: def () -> TestLocation,
    *Ts: Movable & Deinitable
](
    Movable
):
    var funcs: Tuple[*Self.Ts]
    var locator: Self.Locator


struct TestLocation(
    ImplicitlyCopyable,
    Writable
):
    var loc: SourceLocation

    @inline(.always)  # inline to get location of caller
    def __init__(out self):
        self.loc = call_location()

    def filename(self) -> String:

        # show paths relative to the tests folder
        var path = Path(self.loc.file_name())
        var parts = path.parts()
        for i in range(len(parts)):
            if parts[i] == 'cryoluge_tests':
                return "/".join(parts[i+1:])

        # can't relativize the path: just use the whole thing
        return String(path)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.filename())


struct TestSuite(
    Deinitable where False,
    Movable
):
    var _mods: List[_ModuleTests]

    def __init__(out self):
        self._mods = List[_ModuleTests]()

    def add_module[mod: ModuleTests](mut self):
        self._mods.append(_ModuleTests.discover[mod]())

    def run(deinit self):

        var results = List[_ModuleResult]()

        # run all the tests and collect the results
        self._mods.reverse()
        while len(self._mods) > 0:
            var mod = self._mods.pop()
            results.append(mod^.run())

        # need to explicitly deinit the mods list to make the compiler happy,
        # even though the list is empty now
        self._mods^.deinit_with(_ModuleTests.abandon)

        # show the failures
        var num_failures = 0
        for result in results:
            if result.failed():
                num_failures += 1
            print(result)

        print()
        if num_failures > 0:
            print(t" ⛔️ {num_failures} module(s) failed!")
        else:
            print(t" ✅ {len(results)} module(s) SUCCEEDED!")
        print()


@fieldwise_init
struct _ModuleTests(
    Deinitable where False,
    Movable
):
    var _location: TestLocation
    var _suite: Optional[StdTestSuite]
    var _err_discovery: Optional[Error]

    @inline(.always)  # for the call location
    @staticmethod
    def discover[mod: ModuleTests]() -> Self:

        var loc = materialize[mod.locator]()()

        try:
            var suite = StdTestSuite.discover_tests[mod.funcs](location=loc.loc)
            return Self(
                _location = loc,
                _suite = suite^,
                _err_discovery = None
            )
        except e:
            return Self(
                _location = loc,
                _suite = None,
                _err_discovery = e
            )

    def run(deinit self) -> _ModuleResult:

        # check for discovery errors first
        if self._err_discovery is not None:
            self._suite^.deinit_assert_empty()
            return _ModuleResult(self._location, None, self._err_discovery.value())

        var suite = self._suite.take()
        self._suite^.deinit_assert_empty()

        # try to run the tests
        var report: TestSuiteReport
        try:
            report = suite.generate_report()
            suite^.abandon()  # generating the report runs the tests, so we can abandon now
        except e:
            suite^.abandon()
            return _ModuleResult(self._location, None, e)

        # success!
        return _ModuleResult(self._location, report^, None)

    def abandon(deinit self):
        if self._suite is not None:
            self._suite.take().abandon()
        self._suite^.deinit_assert_empty()


@fieldwise_init
struct _ModuleResult(
    Copyable,
    Writable
):
    var location: TestLocation
    var report: Optional[TestSuiteReport]
    var err: Optional[Error]

    def failed(self) -> Bool:
        return self.err is not None
            or (self.report is not None and self.report.value().failures > 0)

    def write_to(self, mut writer: Some[Writer]):
        if self.failed():
            writer.write(" ⛔️ ", self.location)
            if self.err is not None:
                writer.write("\n\t", self.err.value())
            if self.report is not None:
                writer.write("\n", self.report.value())
        else:
            writer.write(" ✅ ", self.location)
