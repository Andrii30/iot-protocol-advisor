"""Entry point.

    python -m protocol_advisor                       launch the GUI
    python -m protocol_advisor --file data.csv        score once, print, exit
    python -m protocol_advisor --file data.csv --watch 300
                                                     re-score every 300s (autonomous)
    python -m protocol_advisor --db "postgresql://u:p@h/db" --query "SELECT ..." --watch 60

Add --out report.csv to append SWITCH recommendations (with a timestamp) each cycle.
Add --switch-threshold P to change how confident a switch must be (default 0.55).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from protocol_advisor.advisor import DEFAULT_SWITCH_THRESHOLD, SWITCH, advise
from protocol_advisor.engine import Engine


def _make_source(args):
    if args.file:
        from protocol_advisor.sources.csv_source import CsvSource

        return CsvSource(args.file)
    from protocol_advisor.sources.sql_source import SqlSource

    return SqlSource(args.query, dsn=args.db)


def _run_once(engine: Engine, source, out: Path | None,
              previous: dict[str, str] | None = None,
              switch_threshold: float = DEFAULT_SWITCH_THRESHOLD) -> dict[str, str]:
    """Score the source once. Print/log only switches new or changed since
    `previous` (device_id -> recommended protocol). Return the current mapping."""
    report = advise(source.load(), engine, switch_threshold=switch_threshold)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    switches = report[report["verdict"] == SWITCH]
    current = dict(zip(switches["device_id"], switches["recommended_protocol"]))

    prev = previous or {}
    changed_ids = [d for d, p in current.items() if prev.get(d) != p]

    counts = report["verdict"].value_counts().to_dict()
    note = f" ({len(changed_ids)} new/changed)" if previous is not None else ""
    agree = report["ml_agrees_rule"].mean() * 100 if len(report) else 0.0
    print(f"[{stamp}] {len(report)} devices — "
          + ", ".join(f"{k}:{v}" for k, v in counts.items())
          + f" · ML/rule-based agreement {agree:.0f}%" + note)

    changed = switches[switches["device_id"].isin(changed_ids)]
    for _, r in changed.iterrows():
        print(f"    SWITCH {r['device_id']}: {r['current_protocol']} -> "
              f"{r['recommended_protocol']} ({r['confidence']:.0%})")

    if out is not None and not changed.empty:
        rows = changed.drop(columns=["probabilities"]).copy()
        rows.insert(0, "checked_at", stamp)
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
    return current


def _headless(args) -> int:
    engine = Engine()
    print("Loading / training model…")
    info = engine.load_or_train()
    print(f"Model: {info.model_name} · macro-F1 {info.macro_f1:.2f}")
    source = _make_source(args)
    out = Path(args.out) if args.out else None
    thr = args.switch_threshold

    if not args.watch:
        _run_once(engine, source, out, switch_threshold=thr)
        return 0

    print(f"Watching every {args.watch}s. Ctrl-C to stop.")
    state: dict[str, str] = {}
    try:
        while True:
            try:
                state = _run_once(engine, source, out, previous=state, switch_threshold=thr)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print(f"    ! {exc}", file=sys.stderr)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="protocol_advisor")
    ap.add_argument("--file", help="input CSV of device measurements")
    ap.add_argument("--db", help="PostgreSQL connection string / URL")
    ap.add_argument("--query", help="SQL query returning the required columns")
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="re-score on this interval instead of exiting")
    ap.add_argument("--out", help="append SWITCH recommendations to this CSV")
    ap.add_argument("--switch-threshold", type=float, default=DEFAULT_SWITCH_THRESHOLD,
                    metavar="P",
                    help=f"min confidence to recommend a switch (default {DEFAULT_SWITCH_THRESHOLD})")
    args = ap.parse_args()

    if args.db and not args.query:
        ap.error("--db requires --query")

    if args.file or args.db:
        return _headless(args)

    # No data source given -> GUI.
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "Tkinter is not available for this Python.\n"
            "  Debian/Ubuntu:  sudo apt install python3-tk\n"
            "  macOS (Homebrew): brew install python-tk\n"
            "  Fedora:         sudo dnf install python3-tkinter\n"
            "Or run headless: python -m protocol_advisor --file <csv>",
            file=sys.stderr,
        )
        return 1

    from protocol_advisor.ui import run

    run(Engine())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
