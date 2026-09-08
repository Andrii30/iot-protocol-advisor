# iot-protocol-advisor

A small desktop app that reads per-device IoT network measurements, uses a
machine-learning model to recommend the best data-transmission protocol for
each device, and gives a plain verdict: **keep the current protocol** or
**switch to protocol X**.

Protocols considered: **MQTT, CoAP, HTTPS, LoRaWAN**.

![screenshot](docs/screenshot.png)
<!-- capture your own: see docs/README.md -->



## How it works

1. **Engine** (`protocol_advisor/engine.py`) trains four scikit-learn
   classifiers — Logistic Regression, Gradient Boosting, Random Forest and a
   small MLP — on labelled network logs, evaluates them on a stratified 25%
   holdout, and keeps the one with the best macro-F1. The fitted model is
   cached to `~/.iot-protocol-advisor/model.pkl` and reused on the next run.
2. **Advisor** (`protocol_advisor/advisor.py`) groups the input rows by
   `device_id`, takes the median of each network feature per device, asks the
   model for the best protocol, and compares it with the device's current
   protocol:
   - recommended == current → `KEEP`
   - recommended != current and confidence ≥ 0.55 → `SWITCH`
   - otherwise → `KEEP_LOW_CONFIDENCE`
3. **UI** (`protocol_advisor/ui.py`) shows one row per device with the verdict,
   confidence and the two most influential features. Double-click a row for the
   full per-protocol probability breakdown.

## Quick start

macOS or Debian/Ubuntu — one command sets up a virtualenv, installs everything
and launches the window:

```bash
./run.sh
```

If it reports Tkinter is missing, run the printed install line
(`brew install python-tk` / `sudo apt install python3-tk`) and rerun `./run.sh`.

## Install (manual)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tkinter ships with CPython on Windows and the python.org macOS builds. If you
get a "Tkinter is not available" message:

- Debian/Ubuntu: `sudo apt install python3-tk`
- macOS + Homebrew Python: `brew install python-tk`
- Fedora: `sudo dnf install python3-tkinter`

## Run

```bash
python -m protocol_advisor
```

On first launch the model trains on the bundled data in
`examples/training_data/` (about a minute), then the window opens. Use
**Open data file…** and point it at a CSV like `examples/sample_devices.csv`.

## Input CSV format

One row per measurement. Rows are grouped by `device_id`, so a device can have
many rows.

| column             | type  | notes                                             |
|--------------------|-------|---------------------------------------------------|
| `device_id`        | str   | device identifier                                 |
| `current_protocol` | str   | one of MQTT, CoAP, HTTPS, LoRaWAN                  |
| `payload_size`     | float | bytes                                             |
| `latency`          | float | ms                                                |
| `jitter`           | float | ms                                                |
| `throughput`       | float | Mbit/s                                            |
| `packet_loss`      | float | fraction or percent — used as given, not rescaled |

Extra columns are ignored. Missing required columns produce a clear error.

## Autonomous / headless mode

Run without the GUI — score once, or keep watching a source on an interval:

```bash
python -m protocol_advisor --file devices.csv                 # score once
python -m protocol_advisor --file devices.csv --watch 300     # every 5 min
python -m protocol_advisor --db "postgresql://u:p@host/db" \
    --query "SELECT ... FROM measurements" --watch 60 --out switches.csv
```

Each cycle prints a summary and every `SWITCH` recommendation. `--out` appends
those rows (with a `checked_at` timestamp) to a CSV, so the tool can run as a
cron job or a systemd service and leave an audit trail.

## Training data and model

The bundled model trains on `examples/training_data/*.csv` — a synthetic set
(50k rows, 5 device archetypes) where the optimal protocol is a transparent
scoring function of the network features plus ~8% label noise. Regenerate it
with `python scripts/generate_dataset.py`.

On that data the holdout (stratified 25% split) is about **macro-F1 0.89 /
accuracy 0.92**, with `HistGradientBoosting` usually winning. The engine also
derives three features from the raw five (`transfer_time`, `burstiness`,
`log_payload`).

Retrain on logs from your own network for real recommendations:

```python
from protocol_advisor import Engine
Engine().retrain("path/to/your/logs/*.csv")
```

Training CSVs need the columns `payload_size, latency, jitter, throughput,
packet_loss, best_protocol` (extra columns are ignored).

## PostgreSQL

Click **Connect to PostgreSQL…**, enter a connection string
(`postgresql://user:pass@host:5432/dbname`) and a query that returns the
columns listed above. The driver comes from `requirements.txt`; for a
packaged install use `pip install "iot-protocol-advisor[db]"`.

For large tables, aggregate in SQL so only ~one row per device is returned:

```sql
SELECT device_id,
       mode() WITHIN GROUP (ORDER BY current_protocol) AS current_protocol,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY payload_size) AS payload_size,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency)      AS latency,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY jitter)       AS jitter,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY throughput)   AS throughput,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY packet_loss)  AS packet_loss
FROM measurements
WHERE ts > now() - interval '1 day'
GROUP BY device_id;
```

## Other data sources

`protocol_advisor/sources/base.py` defines a one-method `DataSource`
interface. To pull from a NetFlow export or an SNMP poll, add a module
implementing `load() -> pandas.DataFrame` with the input columns above.
`CsvSource` and `SqlSource` ship in this release.

## Security note

The trained model is stored as a pickle (`model.pkl`) via `joblib`, which is
standard for scikit-learn. Loading a pickle can execute arbitrary code, so the
app only ever loads a `model.pkl` it wrote itself, under a user-owned
directory. Do not point `PROTOCOL_ADVISOR_HOME` at a directory whose
`model.pkl` came from someone else — retrain instead.

## Tests

```bash
pip install pytest
pytest
```

## License

MIT — see [LICENSE](LICENSE).
