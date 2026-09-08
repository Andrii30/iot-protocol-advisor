from __future__ import annotations

import pandas as pd

from protocol_advisor.__main__ import _run_once
from protocol_advisor.sources.csv_source import CsvSource


def test_run_once_reports_and_appends_switches(engine, devices_frame, tmp_path, capsys):
    csv = tmp_path / "devices.csv"
    devices_frame.to_csv(csv, index=False)
    out = tmp_path / "verdicts.csv"

    n_switch = _run_once(engine, CsvSource(csv), out)

    printed = capsys.readouterr().out
    assert "devices" in printed
    assert n_switch >= 1
    assert out.exists()
    written = pd.read_csv(out)
    assert "checked_at" in written.columns
    assert (written["verdict"] == "SWITCH").all()

    # A second cycle appends, keeping one header.
    _run_once(engine, CsvSource(csv), out)
    assert (pd.read_csv(out)["verdict"] == "SWITCH").all()
    assert len(pd.read_csv(out)) == 2 * len(written)
