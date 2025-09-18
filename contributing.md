
# Contributing

## Building

Build the `.mojopkg` file with the `pixi` task `build_pkg`:
```shell
pixi run build_pkg
```
If successful, the built package will appear at `build/cryoluge.mojopkg`.


## Testing

Run the automated test suite with the `pixi` task:
```shell
pixi run test
```


## IDE

Tragically, [the Mojo extension for PyCharm is abandonware](https://youtrack.jetbrains.com/issue/PY-60990/Mojo-support#focus=Comments-27-12561127.0-0),
so currently the only IDE with decent support for Mojo is VS Code.

To get the Mojo extension's language server to recognize this library,
you'll need to add the absolute path of the `src` folder to the LSP include folders.
And if you want to develop the test suite for the library, you'll want the `test` folder too.

Find the `Mojo › Lsp: Include Dirs` setting in VS Code and add the absolute path to the `src` folder.
On your computer, the path may look something like this:
```
/home/me/projects/cryoluge/src
```
And test `test` folder path may look something like this:
```
/home/me/projects/cryoluge/test
```

Alternatively, you can add the following JSON config to your `~/.Config/Code/User/settings.json` file to configure the setting in JSON:
```json
{
    "mojo.lsp.includeDirs": [
        "/home/jeff/cryoluge/src", 
        "/home/jeff/cryoluge/test"
    ]
}
```

Sadly, `mojo.lsp.includeDirs` does not appear to be recognized at the workspace level,
so there's no good way to get the language server to auto-include this library
(by, say, putting a relative path in the workspace's `settings.json` file),
even if you're working on this project directly in VS Code.
