# Contributing

Thanks for taking a look.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Before opening a PR

- `pytest -q` passes (CI runs it on Python 3.10 and 3.12).
- New behaviour has a test. Keep tests fast — no network, no real database
  (use the sqlite fixture pattern in `tests/test_sql_source.py`).
- Match the surrounding style: type hints, short docstrings, module-level
  constants. No new runtime dependencies without a good reason.
- One focused change per PR.

## Adding a data source

Implement `DataSource.load()` in a new module under `protocol_advisor/sources/`
returning a DataFrame with the columns in `sources/base.py::INPUT_COLUMNS`.
Export it from `sources/__init__.py` and add a test.

## Commit messages and releases

Commits on `main` use [Conventional Commits](https://www.conventionalcommits.org/):

- `fix: …` — a bug fix (bumps the patch version)
- `feat: …` — a new feature (bumps the minor version)
- `feat!: …` or a `BREAKING CHANGE:` footer — a breaking change (bumps major)
- `docs:`, `test:`, `ci:`, `chore:`, `refactor:` — no version bump

On every push to `main`, release-please opens or updates a **Release PR** that
bumps the version in `pyproject.toml` and `protocol_advisor/__init__.py` and
writes the `CHANGELOG.md` entry. Merging that PR tags the release and publishes
it. Don't edit the version by hand.

## Reporting bugs

Open an issue with the command you ran, what happened, and what you expected.
Include the model line from the status bar / CLI output.
