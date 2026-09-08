from __future__ import annotations

import pandas as pd

from protocol_advisor.__main__ import _run_once
from protocol_advisor.sources.csv_source import CsvSource


def test_run_once_reports_and_appends_switches(engine, devices_frame, tmp_path, capsys):
    csv = tmp_path / "devices.csv"
    devices_frame.to_csv(csv, index=False)
    out = tmp_path / "verdicts.csv"

    state = _run_once(engine, CsvSource(csv), out)

    printed = capsys.readouterr().out
    assert "devices" in printed
    assert len(state) >= 1  # at least one device recommended a switch
    assert out.exists()
    written = pd.read_csv(out)
    assert "checked_at" in written.columns
    assert (written["verdict"] == "SWITCH").all()
    n_first = len(written)

    # Second cycle with the same state -> nothing new, no rows appended.
    _run_once(engine, CsvSource(csv), out, previous=state)
    assert len(pd.read_csv(out)) == n_first
    assert "0 new/changed" in capsys.readouterr().out


def test_switch_threshold_above_one_never_switches(engine, devices_frame, tmp_path, capsys):
    csv = tmp_path / "devices.csv"
    devices_frame.to_csv(csv, index=False)
    state = _run_once(engine, CsvSource(csv), None, switch_threshold=1.01)
    assert state == {}  # nothing clears the bar
