
struct LexicalScope:
    """
    For when you want an extra lexical scope to prevent variables from leaking into unwanted places.

    ```mojo
    from cryoluge.lang import LexicalScope

    with LexicalScope():
        var foo = 1
    print(foo) # compiler error! (desired)
    ```

    A future version of Mojo may implement extra lexical scopes at a language level, see:
    https://github.com/modular/modular/issues/5371
    """

    fn __init__(out self):
        pass

    fn __enter__(self):
        pass

    fn __exit__(self):
        pass
