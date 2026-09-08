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

## Reporting bugs

Open an issue with the command you ran, what happened, and what you expected.
Include the model line from the status bar / CLI output.
