
from std.builtin.error import StackTrace
from std.builtin.debug_assert import _assert_enabled, ASSERT_MODE
from std.os import abort


@inline(.always)
def debug_assert_with_stack[
    assert_mode: StaticString = "none",
    *Ts: Writable,
    cpu_only: Bool = False,
](cond: Bool, *messages: *Ts):
    """
    Requires setting environment variable MOJO_ENABLE_STACK_TRACE_ON_ERROR on the runtime executable.
    Set it to some non-empty string, like '1' or 'yup'.
    Also, adjust build settings to add source information to debug symbols, eg:
      -debug-level=full
      -debug-level=line-tables
    Need a sentence here to make the (overly pedantic) compiler happy.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():

        if cond:
            return

        # build the message
        var msg = String()
        comptime for i in range(messages.__len__()):
            msg += String(messages[i])

        var stack = StackTrace.collect_if_enabled(depth=0)
        if stack is not None:
            msg += "\n"
            msg += String(stack.value())

        print("Assert Error: ", msg, sep="")

        comptime if ASSERT_MODE != "warn":
            abort()
