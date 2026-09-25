
from builtin.error import StackTrace
from builtin.debug_assert import _assert_enabled, ASSERT_MODE
from os import abort


@always_inline
fn stack_trace() -> String:
    """
    Requires setting environment variable MOJO_ENABLE_STACK_TRACE_ON_ERROR on the runtime executable.
    Set it to some non-empty string, like '1' or 'yup'.
    Also, adjust build settings to add source information to debug symbols, eg:
      -debug-level=full
      -debug-level=line-tables
    Need a sentence here to make the (overly pedantic) compiler happy.
    """
    return String(StackTrace(depth=0))


@always_inline
fn debug_assert_with_stack[
    assert_mode: StaticString = "none",
    *Ts: Writable,
    cpu_only: Bool = False,
](cond: Bool, *messages: *Ts):

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():

        if cond:
            return

        # build the message
        var msg = String()
        @parameter
        for i in range(messages.__len__()):
            msg += String(messages[i])

        print(
            "Assert Error: ", msg,
            "\n", stack_trace(),
            sep=""
        )

        @parameter
        if ASSERT_MODE != "warn":
            abort()
