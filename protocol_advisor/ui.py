"""Tkinter desktop UI.

A window with a menu bar, a toolbar, a filter/search row, a results table
(one row per device, tinted by verdict) and a status bar. Training and
scoring run on a worker thread; results come back to the Tk thread through a
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
from protocol_advisor.sources.sql_source import SqlSource

_COLUMNS = [
    ("device_id", "Device", 150, "w"),
    ("current_protocol", "Current", 90, "center"),
    ("recommended_protocol", "Recommended", 110, "center"),
    ("confidence", "Confidence", 95, "center"),
    ("verdict", "Verdict", 165, "w"),
    ("rule_based", "Rule-based", 95, "center"),
    ("agree", "ML=rule", 75, "center"),
    ("n_samples", "Samples", 70, "center"),
    ("factor_1", "Top factor", 150, "w"),
    ("factor_2", "2nd factor", 150, "w"),
]

_VERDICT_ROW_BG = {
    KEEP: "#e9f6ec",
    SWITCH: "#fff1e2",
    KEEP_LOW_CONFIDENCE: "#f1f1f3",
}
_VERDICT_FG = {KEEP: "#1b7f2e", SWITCH: "#d2691e", KEEP_LOW_CONFIDENCE: "#606060"}
_FILTERS = ["All", "Switch", "Keep", "Low-confidence"]
_FILTER_VERDICT = {"Switch": SWITCH, "Keep": KEEP, "Low-confidence": KEEP_LOW_CONFIDENCE}


def _csv_safe(value: object) -> str:
    """Neutralise spreadsheet formula injection in passthrough text cells."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


class AdvisorApp:
    def __init__(self, root: tk.Tk, engine: Engine):
        self.root = root
        self.engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._devices_df: pd.DataFrame | None = None
        self._result_df: pd.DataFrame | None = None
        self._view_df: pd.DataFrame | None = None
        self._sort_asc: dict[str, bool] = {}
        self._filter = tk.StringVar(value="All")
        self._search = tk.StringVar(value="")

        root.title("IoT Protocol Advisor")
        root.geometry("1180x640")
        root.minsize(900, 480)
        self._init_style()
        self._build_menu()
        self._build_widgets()
        self.root.after(100, self._poll_queue)

    # -- style ---------------------------------------------------------

    def _init_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Helvetica",12))
        style.configure("Treeview", rowheight=27, font=("Helvetica",12), borderwidth=0)
        style.configure("Treeview.Heading", font=("Helvetica",11, "bold"), padding=4)
        style.map("Treeview", background=[("selected", "#3a7bd5")],
                  foreground=[("selected", "white")])
        style.configure("Toolbutton", padding=6)
        style.configure("Summary.TLabel", font=("Helvetica",15))
        style.configure("Status.TLabel", font=("Helvetica",10), foreground="#555")

    def _build_menu(self) -> None:
        m = tk.Menu(self.root)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Open data file…", command=self._open_file, accelerator="Cmd+O")
        filem.add_command(label="Connect to PostgreSQL…", command=self._connect_db)
        filem.add_separator()
        filem.add_command(label="Export report…", command=self._export, accelerator="Cmd+E")
        filem.add_separator()
        filem.add_command(label="Quit", command=self.root.destroy)
        m.add_cascade(label="File", menu=filem)

        modelm = tk.Menu(m, tearoff=0)
        modelm.add_command(label="Retrain model…", command=self._retrain)
        m.add_cascade(label="Model", menu=modelm)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About", command=self._about)
        m.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=m)
        self.root.bind("<Command-o>", lambda _e: self._open_file())
        self.root.bind("<Command-e>", lambda _e: self._export())

    def _build_widgets(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")
        self.btn_open = ttk.Button(bar, text="Open data file…", style="Toolbutton",
                                   command=self._open_file)
        self.btn_open.pack(side="left")
        self.btn_db = ttk.Button(bar, text="Connect to PostgreSQL…", style="Toolbutton",
                                 command=self._connect_db)
        self.btn_db.pack(side="left", padx=(6, 0))
        self.btn_retrain = ttk.Button(bar, text="Retrain model", style="Toolbutton",
                                      command=self._retrain)
        self.btn_retrain.pack(side="left", padx=(6, 0))
        self.btn_export = ttk.Button(bar, text="Export report…", style="Toolbutton",
                                     command=self._export, state="disabled")
        self.btn_export.pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)

        summ = ttk.Frame(self.root, padding=(12, 2))
        summ.pack(fill="x")
        self.summary = tk.StringVar(value="Open a data file or connect to PostgreSQL to begin.")
        ttk.Label(summ, textvariable=self.summary, style="Summary.TLabel").pack(side="left")

        filt = ttk.Frame(self.root, padding=(12, 4))
        filt.pack(fill="x")
        ttk.Label(filt, text="Show:").pack(side="left")
        for name in _FILTERS:
            ttk.Radiobutton(filt, text=name, value=name, variable=self._filter,
                            command=self._apply_filter).pack(side="left", padx=(4, 0))
        ttk.Label(filt, text="   Search:").pack(side="left")
        ent = ttk.Entry(filt, textvariable=self._search, width=22)
        ent.pack(side="left", padx=(4, 0))
        self._search.trace_add("write", lambda *_: self._apply_filter())

        wrap = ttk.Frame(self.root, padding=(10, 4))
        wrap.pack(fill="both", expand=True)
        cols = [c[0] for c in _COLUMNS]
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        for key, label, width, anchor in _COLUMNS:
            self.tree.heading(key, text=label, command=lambda k=key: self._sort_by(k))
            self.tree.column(key, width=width, anchor=anchor, stretch=(key in ("factor_1", "factor_2")))
        for verdict, bg in _VERDICT_ROW_BG.items():
            self.tree.tag_configure(verdict, background=bg, foreground=_VERDICT_FG[verdict])
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._on_row_double_click)

        self.status = tk.StringVar(value="No model loaded.")
        ttk.Label(self.root, textvariable=self.status, style="Status.TLabel",
                  relief="sunken", anchor="w", padding=4).pack(fill="x", side="bottom")

    # -- model status --------------------------------------------------------

    def refresh_status(self) -> None:
        try:
            info = self.engine.info
        except RuntimeError:
            self.status.set("No model loaded.")
            return
        warn = "  ⚠ low-confidence evaluation (little training data)" if info.low_confidence_eval else ""
        self.status.set(
            f"Model: {info.model_name} · holdout macro-F1 {info.macro_f1:.2f} · "
            f"accuracy {info.accuracy:.2f} · trained {info.trained_at} · "
            f"sklearn {info.sklearn_version}{warn}"
        )

    # -- async plumbing --------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.btn_open, self.btn_db, self.btn_retrain):
            b.config(state=state)
        self.btn_export.config(
            state="disabled" if (busy or self._result_df is None) else "normal"
        )
        if busy:
            self.progress.pack(side="right")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
        self.root.config(cursor="watch" if busy else "")

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
            try:
                on_success, result, error = self._queue.get_nowait()
            except queue.Empty:
                return
            self._set_busy(False)
            if error is not None:
                self.refresh_status()
                messagebox.showerror("IoT Protocol Advisor", str(error))
            else:
                try:
                    on_success(result)
                except Exception as exc:  # noqa: BLE001 - keep the poller alive
                    self.refresh_status()
                    messagebox.showerror("IoT Protocol Advisor", str(exc))
        finally:
            self.root.after(100, self._poll_queue)

    # -- actions --------------------------------------------------------

    def _ingest(self, source, origin: str) -> None:
        def work():
            devices = source.load()
            return devices, advise(devices, self.engine)

        def done(result):
            self._devices_df, self._result_df = result
            self._sort_asc.clear()
            self._origin = origin
            self._apply_filter()
            self.btn_export.config(state="normal")
            self.status.set(f"Loaded {origin}.")

        self._run_async(work, done)

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open device measurements CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._ingest(CsvSource(path), Path(path).name)

    def _connect_db(self) -> None:
        dsn, query = self._ask_db_query()
        if dsn and query:
            self._ingest(SqlSource(query, dsn=dsn), "PostgreSQL")

    def _ask_db_query(self) -> tuple[str | None, str | None]:
        dlg = tk.Toplevel(self.root)
        dlg.title("Connect to PostgreSQL")
        dlg.transient(self.root)
        dlg.grab_set()
        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Connection string (libpq / URL):").pack(anchor="w")
        dsn_var = tk.StringVar(value="postgresql://user:password@localhost:5432/dbname")
        ttk.Entry(frame, textvariable=dsn_var, width=64).pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="SQL query — must return columns:\n"
                  "device_id, current_protocol, payload_size, latency, jitter, "
                  "throughput, packet_loss").pack(anchor="w")
        query_txt = tk.Text(frame, width=64, height=10)
        query_txt.pack(fill="both", expand=True, pady=(0, 8))
        query_txt.insert("1.0", "SELECT device_id, current_protocol, payload_size,\n"
                                "       latency, jitter, throughput, packet_loss\n"
                                "FROM measurements;")

        result: dict[str, str] = {}

        def ok():
            result["dsn"] = dsn_var.get().strip()
            result["query"] = query_txt.get("1.0", "end").strip()
            dlg.destroy()

        btns = ttk.Frame(frame)
        btns.pack(fill="x")
        ttk.Button(btns, text="Load", command=ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 6))

        self.root.wait_window(dlg)
        return result.get("dsn"), result.get("query")

    def _retrain(self) -> None:
        folder = filedialog.askdirectory(title="Folder of training CSV files")
        if not folder:
            return
        glob = str(Path(folder) / "*.csv")

        def rescore(_info):
            self.refresh_status()
            if self._devices_df is not None:
                self._run_async(lambda: advise(self._devices_df, self.engine),
                                self._apply_rescore)

        self._run_async(lambda: self.engine.retrain(glob), rescore)
        self.status.set("Retraining model…")

    def _apply_rescore(self, report: pd.DataFrame) -> None:
        self._result_df = report
        self._sort_asc.clear()
        self._apply_filter()

    def _export(self) -> None:
        if self._view_df is None or self._view_df.empty:
            return
        path = filedialog.asksaveasfilename(
            title="Export report", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                self._view_df.to_json(path, orient="records", indent=2)
            else:
                flat = self._view_df.drop(columns=["probabilities"]).copy()
                for col in ("device_id", "current_protocol", "recommended_protocol", "rule_based"):
                    flat[col] = flat[col].map(_csv_safe)
                flat.to_csv(path, index=False)
        except OSError as exc:
            messagebox.showerror("IoT Protocol Advisor", f"Could not write file:\n{exc}")
            return
        self.status.set(f"Report written to {path}")

    def _about(self) -> None:
        messagebox.showinfo(
            "IoT Protocol Advisor",
            "Recommends the optimal IoT data protocol per device "
            "(MQTT / CoAP / HTTPS / LoRaWAN) from network measurements.\n\n"
            "MIT licensed · github.com/Andrii30/iot-protocol-advisor",
        )

    # -- table --------------------------------------------------------

    def _apply_filter(self) -> None:
        if self._result_df is None:
            return
        df = self._result_df
        choice = self._filter.get()
        if choice in _FILTER_VERDICT:
            df = df[df["verdict"] == _FILTER_VERDICT[choice]]
        q = self._search.get().strip().lower()
        if q:
            df = df[df["device_id"].astype(str).str.lower().str.contains(q, regex=False)]
        self._view_df = df
        self._populate(df)
        total = len(self._result_df)
        shown = len(df)
        seen = f"{shown} of {total}" if shown != total else f"{total}"
        origin = getattr(self, "_origin", "data")
        self.summary.set(f"{seen} devices from {origin}   ·   "
                         + self._verdict_summary(self._result_df))

    def _populate(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, row in df.iterrows():
            values = [
                row["device_id"], row["current_protocol"], row["recommended_protocol"],
                f"{row['confidence']:.2f}", row["verdict"], row["rule_based"],
                "✓" if row["ml_agrees_rule"] else "✗",
                row["n_samples"], row["factor_1"], row["factor_2"],
            ]
            self.tree.insert("", "end", values=values, tags=(row["verdict"],))

    def _sort_by(self, key: str) -> None:
        if self._view_df is None:
            return
        col = {"agree": "ml_agrees_rule"}.get(key, key)
        ascending = self._sort_asc.get(key, True)
        self._view_df = self._view_df.sort_values(col, ascending=ascending, kind="stable")
        self._sort_asc[key] = not ascending
        self._populate(self._view_df)

    def _on_row_double_click(self, _event) -> None:
        item = self.tree.focus()
        if not item or self._view_df is None:
            return
        device_id = self.tree.item(item, "values")[0]
        row = self._view_df[self._view_df["device_id"].astype(str) == str(device_id)]
        if not row.empty:
            self._show_detail(row.iloc[0])

    def _show_detail(self, row: pd.Series) -> None:
        top = tk.Toplevel(self.root)
        top.title(f"Device {row['device_id']}")
        top.geometry("440x460")
        frame = ttk.Frame(top, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=str(row["verdict"]), font=("Helvetica",15, "bold"),
                  foreground=_VERDICT_FG.get(row["verdict"], "#222")).pack(anchor="w")
        ttk.Label(frame, text=f"Current: {row['current_protocol']}      "
                  f"Recommended: {row['recommended_protocol']}  ({row['confidence']:.0%})"
                  ).pack(anchor="w")
        agree = "agrees with" if row.get("ml_agrees_rule") else "differs from"
        ttk.Label(frame, text=f"Rule-based baseline: {row.get('rule_based', '—')}  "
                  f"({agree} the model)").pack(anchor="w", pady=(0, 10))

        ttk.Label(frame, text="Protocol probabilities", font=("Helvetica",11, "bold")).pack(anchor="w")
        probs = dict(row["probabilities"])
        for proto in sorted(probs, key=probs.get, reverse=True):
            line = ttk.Frame(frame)
            line.pack(fill="x", pady=1)
            ttk.Label(line, text=proto, width=10).pack(side="left")
            ttk.Progressbar(line, maximum=1.0, value=probs[proto], length=230).pack(side="left", padx=6)
            ttk.Label(line, text=f"{probs[proto]:.1%}").pack(side="left")

        ttk.Label(frame, text="Median conditions", font=("Helvetica",11, "bold")).pack(anchor="w", pady=(12, 0))
        if self._devices_df is not None:
            grp = self._devices_df[
                self._devices_df["device_id"].astype(str) == str(row["device_id"])
            ]
            medians = grp[FEATURES].median()
            for feat in FEATURES:
                ttk.Label(frame, text=f"{feat}: {medians[feat]:.3g}").pack(anchor="w")

    @staticmethod
    def _verdict_summary(df: pd.DataFrame) -> str:
        c = df["verdict"].value_counts()
        parts = [f"{v}: {c[v]}" for v in (SWITCH, KEEP, KEEP_LOW_CONFIDENCE) if v in c]
        agree = df["ml_agrees_rule"].mean() * 100 if len(df) else 0
        return "  ·  ".join(parts) + f"   ·   ML=rule {agree:.0f}%"


def run(engine: Engine, autotrain: bool = True) -> None:
    root = tk.Tk()
    app = AdvisorApp(root, engine)
    if autotrain:
        app.status.set("Loading / training model… (first run may take a minute)")
        app._run_async(engine.load_or_train, lambda _i: app.refresh_status())
    else:
        app.refresh_status()
    root.mainloop()
