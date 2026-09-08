# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-09-08

Initial release.

### Added
- Engine: trains Logistic Regression, Gradient Boosting, Random Forest,
  HistGradientBoosting and an MLP; keeps the best by macro-F1 on a stratified
  holdout; caches to `model.pkl` with a schema version and the scikit-learn
  version it was trained with.
- Three derived features (`transfer_time`, `burstiness`, `log_payload`).
- Advisor: per-device median aggregation, `KEEP` / `SWITCH` /
  `KEEP_LOW_CONFIDENCE` verdict, plus a rule-based baseline column and
  ML-vs-rule agreement.
- Data sources: CSV file/directory and PostgreSQL (`SqlSource`, psycopg 3).
- Tkinter desktop UI: menu bar, verdict/search filters, results table tinted
  by verdict, detail popup, retrain, export, training progress bar.
- Headless mode: `--file` / `--db` / `--query`, `--watch` interval,
  `--out` audit CSV that logs only new or changed switch recommendations.
- `run.sh` one-command launcher (macOS / Debian).
- systemd unit and cron examples under `deploy/`.
- Synthetic dataset generator (`scripts/generate_dataset.py`).
- 32 tests (including headless UI smoke tests), CI on Python 3.10 and 3.12
  under xvfb.
