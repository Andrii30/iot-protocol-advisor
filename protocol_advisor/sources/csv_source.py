"""Load measurements from a CSV file or a directory of CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from protocol_advisor.sources.base import INPUT_COLUMNS, DataSource


class CsvLoadError(RuntimeError):
    """Raised when the CSV path is unusable or has no valid rows."""


class CsvSource(DataSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise CsvLoadError(f"Path not found: {self.path}")

        if self.path.is_dir():
            files = sorted(self.path.glob("*.csv"))
            if not files:
                raise CsvLoadError(f"No .csv files in directory: {self.path}")
        else:
            files = [self.path]

        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as exc:  # noqa: BLE001 - clean message for the UI
                raise CsvLoadError(f"Cannot read {f}: {exc}") from exc
            df["source_file"] = f.name
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)

        missing = [c for c in INPUT_COLUMNS if c not in data.columns]
        if missing:
            raise CsvLoadError(
                "Input file is missing required columns: " + ", ".join(missing)
            )

        data = data.dropna(subset=INPUT_COLUMNS)
        if data.empty:
            raise CsvLoadError("No rows with all required columns present.")
        return data
