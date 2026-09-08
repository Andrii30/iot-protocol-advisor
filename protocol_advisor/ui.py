"""Tkinter desktop UI.

One window: a toolbar (open data / retrain / export), a table with one row
per device, and a status bar showing the active model. Training and scoring
run on a worker thread; results are handed back to the Tk thread through a
queue polled with ``after()`` so the event loop never blocks.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from protocol_advisor.advisor import KEEP, KEEP_LOW_CONFIDENCE, SWITCH, advise
from protocol_advisor.engine import FEATURES, Engine
from protocol_advisor.sources.csv_source import CsvSource

_COLUMNS = [
    ("device_id", "Device", 140),
    ("current_protocol", "Current", 90),
    ("recommended_protocol", "Recommended", 110),
    ("confidence", "Confidence", 90),
    ("verdict", "Verdict", 170),
    ("n_samples", "Samples", 70),
    ("factor_1", "Factor 1", 150),
    ("factor_2", "Factor 2", 150),
]

def _csv_safe(value: object) -> str:
    """Neutralise spreadsheet formula injection in passthrough text cells."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


_VERDICT_COLOUR = {
    KEEP: "#1b5e20",
    SWITCH: "#e65100",
    KEEP_LOW_CONFIDENCE: "#616161",
}


class AdvisorApp:
    def __init__(self, root: tk.Tk, engine: Engine):
        self.root = root
        self.engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._devices_df: pd.DataFrame | None = None
        self._result_df: pd.DataFrame | None = None
        self._sort_asc: dict[str, bool] = {}

        root.title("IoT Protocol Advisor")
        root.geometry("1024x600")
        self._build_widgets()
        self.root.after(100, self._poll_queue)

    # -- layout ----------------------------------------------------------

    def _build_widgets(self) -> None:
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill="x")
        self.btn_open = ttk.Button(bar, text="Open data file…", command=self._open_file)
        self.btn_open.pack(side="left")
        self.btn_retrain = ttk.Button(bar, text="Retrain model", command=self._retrain)
        self.btn_retrain.pack(side="left", padx=(6, 0))
        self.btn_export = ttk.Button(
            bar, text="Export report…", command=self._export, state="disabled"
        )
        self.btn_export.pack(side="left", padx=(6, 0))

        cols = [c[0] for c in _COLUMNS]
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", selectmode="browse")
        for key, label, width in _COLUMNS:
            self.tree.heading(key, text=label, command=lambda k=key: self._sort_by(k))
            self.tree.column(key, width=width, anchor="w")
        for verdict, colour in _VERDICT_COLOUR.items():
            self.tree.tag_configure(verdict, foreground=colour)
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tree.bind("<Double-1>", self._on_row_double_click)

        self.status = tk.StringVar(value="No model loaded.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=4).pack(fill="x", side="bottom")

    # -- model status --------------------------------------------------------

    def refresh_status(self) -> None:
        try:
            info = self.engine.info
        except RuntimeError:
            self.status.set("No model loaded.")
            return
        warn = "  ⚠ low-confidence evaluation (little training data)" if info.low_confidence_eval else ""
        self.status.set(
            f"Model: {info.model_name} · holdout macro-F1: {info.macro_f1:.2f} · "
            f"accuracy: {info.accuracy:.2f} · trained: {info.trained_at}{warn}"
        )

    # -- async plumbing --------------------------------------------------------

    def _set_busy(self, busy: bool, note: str | None = None) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.btn_open.config(state=state)
        self.btn_retrain.config(state=state)
        self.btn_export.config(
            state="disabled" if (busy or self._result_df is None) else "normal"
        )
        self.root.config(cursor="watch" if busy else "")
        if note:
            self.status.set(note)

    def _run_async(self, work, on_success) -> None:
        if self._busy:
            return
        self._set_busy(True)

        def runner():
            try:
                self._queue.put((on_success, work(), None))
            except Exception as exc:  # noqa: BLE001 - reported in the UI thread
                self._queue.put((on_success, None, exc))

        threading.Thread(target=runner, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            on_success, result, error = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            if error is not None:
                self.refresh_status()
                messagebox.showerror("IoT Protocol Advisor", str(error))
            else:
                on_success(result)
        self.root.after(100, self._poll_queue)

    # -- actions --------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open device measurements CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        def work():
            devices = CsvSource(path).load()
            report = advise(devices, self.engine)
            return devices, report

        def done(result):
            self._devices_df, self._result_df = result
            self._populate(self._result_df)
            self.btn_export.config(state="normal")
            self.status.set(
                f"{len(self._result_df)} device(s) from {Path(path).name} · "
                + self._verdict_summary(self._result_df)
            )

        self._run_async(work, done)

    def _retrain(self) -> None:
        folder = filedialog.askdirectory(title="Folder of training CSV files")
        if not folder:
            return
        glob = str(Path(folder) / "*.csv")

        def done(_info):
            self.refresh_status()
            # Re-score the loaded file against the new model, if any.
            if self._devices_df is not None:
                self._result_df = advise(self._devices_df, self.engine)
                self._populate(self._result_df)

        self._run_async(lambda: self.engine.retrain(glob), done)
        self.status.set("Retraining model…")

    def _export(self) -> None:
        if self._result_df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export report",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                self._result_df.to_json(path, orient="records", indent=2)
            else:
                flat = self._result_df.drop(columns=["probabilities"]).copy()
                for col in ("device_id", "current_protocol", "recommended_protocol"):
                    flat[col] = flat[col].map(_csv_safe)
                flat.to_csv(path, index=False)
        except OSError as exc:
            messagebox.showerror("IoT Protocol Advisor", f"Could not write file:\n{exc}")
            return
        self.status.set(f"Report written to {path}")

    # -- table --------------------------------------------------------

    def _populate(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, row in df.iterrows():
            values = [
                row["device_id"],
                row["current_protocol"],
                row["recommended_protocol"],
                f"{row['confidence']:.2f}",
                row["verdict"],
                row["n_samples"],
                row["factor_1"],
                row["factor_2"],
            ]
            self.tree.insert("", "end", values=values, tags=(row["verdict"],))

    def _sort_by(self, key: str) -> None:
        if self._result_df is None:
            return
        ascending = self._sort_asc.get(key, True)
        self._result_df = self._result_df.sort_values(key, ascending=ascending, kind="stable")
        self._sort_asc[key] = not ascending
        self._populate(self._result_df)

    def _on_row_double_click(self, _event) -> None:
        item = self.tree.focus()
        if not item or self._result_df is None:
            return
        device_id = self.tree.item(item, "values")[0]
        row = self._result_df[self._result_df["device_id"].astype(str) == str(device_id)]
        if row.empty:
            return
        self._show_detail(row.iloc[0])

    def _show_detail(self, row: pd.Series) -> None:
        top = tk.Toplevel(self.root)
        top.title(f"Device {row['device_id']}")
        top.geometry("420x420")
        frame = ttk.Frame(top, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=f"Verdict: {row['verdict']}", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=f"Current: {row['current_protocol']}    "
            f"Recommended: {row['recommended_protocol']} ({row['confidence']:.0%})",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(frame, text="Protocol probabilities", font=("", 10, "bold")).pack(anchor="w")
        probs: dict[str, float] = dict(row["probabilities"])
        for proto in sorted(probs, key=probs.get, reverse=True):
            line = ttk.Frame(frame)
            line.pack(fill="x")
            ttk.Label(line, text=proto, width=10).pack(side="left")
            bar = ttk.Progressbar(line, maximum=1.0, value=probs[proto], length=220)
            bar.pack(side="left", padx=6)
            ttk.Label(line, text=f"{probs[proto]:.1%}").pack(side="left")

        ttk.Label(frame, text="Median conditions", font=("", 10, "bold")).pack(anchor="w", pady=(10, 0))
        if self._devices_df is not None:
            grp = self._devices_df[
                self._devices_df["device_id"].astype(str) == str(row["device_id"])
            ]
            medians = grp[FEATURES].median()
            for feat in FEATURES:
                ttk.Label(frame, text=f"{feat}: {medians[feat]:.3g}").pack(anchor="w")

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _verdict_summary(df: pd.DataFrame) -> str:
        counts = df["verdict"].value_counts()
        return ", ".join(f"{v}: {counts[v]}" for v in counts.index)


def run(engine: Engine, autotrain: bool = True) -> None:
    root = tk.Tk()
    app = AdvisorApp(root, engine)

    if autotrain:
        app.status.set("Loading / training model… (first run may take a minute)")
        app._run_async(engine.load_or_train, lambda _info: app.refresh_status())
    else:
        app.refresh_status()

    root.mainloop()
