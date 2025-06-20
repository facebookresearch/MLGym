# Contributing to <span style="font-variant:small-caps;">MLGym</span>

We want to make contributing to this project as easy and transparent as
possible.

## Development

We use ruff as linter/formatter and mypy for type checking.

1. Install the developmental dependencies

    `pip install -e .[dev]`
2. Install mypy and ruff extensions in your favorite code editor.

3. Here are some design choices to keep in mind before you start developing
   - **Type Hints**: All functions must have complete type annotations using native Python types (minimal imports from `typing` library)
   - **Docstrings**: Use Google-style docstrings for all functions with proper Args/Returns/Raises sections
   - **Error Handling**: Implement proper exception handling with specific error types
   - **Path Handling**: Use `pathlib.Path` instead of `os.path` for all file/path operations
   - **Code Style**: Follow PEP 8 guidelines and use f-strings for string formatting

For VS Code and Cursor, we use the following workspace settings to streamline the development. This is just one option, and you are free to use your own development config.

```json
"mypy-type-checker.importStrategy": "fromEnvironment",
"mypy-type-checker.reportingScope": "workspace",
"mypy-type-checker.preferDaemon": true,
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.formatOnSaveMode": "file",
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    },
},
"ruff.nativeServer": "on",
"ruff.enable": true,
"ruff.lint.enable": true,
"ruff.organizeImports": true,
"ruff.fixAll": true,
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Make sure your code lints.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License

By contributing to <span style="font-variant:small-caps;">MLGym</span>, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
