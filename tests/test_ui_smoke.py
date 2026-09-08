"""Headless smoke test for the Tkinter UI.

Skips when Tk can't initialise (CI without a display / without python3-tk).
It drives the app the way the worker-thread callbacks would, minus threads.
"""

from __future__ import annotations

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(engine, devices_frame):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    from protocol_advisor.advisor import advise
    from protocol_advisor.ui import AdvisorApp

    a = AdvisorApp(root, engine)
    report = advise(devices_frame, engine)
    a._devices_df, a._result_df = devices_frame, report
    a._origin = "test.csv"
    a._apply_filter()
    yield a
    root.destroy()


def test_table_populates_and_summary_set(app):
    assert len(app.tree.get_children()) == 2
    assert "devices from test.csv" in app.summary.get()


def test_verdict_filter(app):
    app._filter.set("Switch")
    app._apply_filter()
    n_switch = (app._result_df["verdict"] == "SWITCH").sum()
    assert len(app.tree.get_children()) == n_switch

    app._filter.set("All")
    app._apply_filter()
    assert len(app.tree.get_children()) == 2


def test_search_filter(app):
    app._search.set("dev-https")
    app._apply_filter()
    assert len(app.tree.get_children()) == 1


def test_sort_does_not_crash_including_synthetic_agree_column(app):
    for key in ("confidence", "device_id", "agree", "verdict"):
        app._sort_by(key)
        app._sort_by(key)
    assert len(app.tree.get_children()) == 2


def test_detail_popup_builds(app):
    app._show_detail(app._view_df.iloc[0])
    # a Toplevel now exists as a child of root
    assert any(isinstance(w, tk.Toplevel) for w in app.root.winfo_children())


def test_export_csv_sanitises(app, tmp_path):
    app._view_df = app._result_df
    out = tmp_path / "r.csv"
    app._view_df.drop(columns=["probabilities"]).to_csv(out, index=False)
    assert pd.read_csv(out).shape[0] == 2
