# Contributing to EarthRS

Please fork the repository if you would like to contribute. Use branch names that are descriptive of the change. 

## Development setup

```bash
pip install -e .[dev]
```

Run the test suite and linter before opening a pull request:

```bash
pytest
ruff check .
```

## Documentation

User-facing documentation lives in [docs/](docs/) and is built with MkDocs. If you change or add public behaviour, update the relevant page there.

```bash
pip install -e .[docs]
mkdocs serve
```

`mkdocs serve` runs a local preview at `http://127.0.0.1:8000` that rebuilds as you edit. Docs are built with `mkdocs build --strict`, so broken links or nav references will fail CI.

## Style

- Use British English spelling in user-facing text and documentation (for example,
  "visualisation", "normalisation", "optimisation").
- Keep APIs clean, typed, and composable.
- `Scene`, `Dataset`, and `Samples` are immutable — methods that change state should return a new instance rather than mutating in place, consistent with the rest of the codebase.
- Prefer small, focused changes aligned with existing project structure.

## Pull requests

- Keep PRs focused on a single change; open separate PRs for unrelated fixes.
- Add or update tests for behaviour you add or change.
- Make sure `pytest` and `ruff check .` pass locally before requesting review.
- CI runs the same checks (tests across Python 3.10–3.12, lint, and a strict docs build) on every push and pull request.

## Reporting issues

Use the bug report or feature request templates under **New issue** on GitHub.
