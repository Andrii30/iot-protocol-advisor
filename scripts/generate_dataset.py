"""Generate a large, principled synthetic dataset for the protocol advisor.

The optimal protocol for a measurement is decided by a transparent scoring
function over the five network features (latency, jitter, packet loss,
throughput, payload size) plus ~8% label noise. Because the target has real
structure, models trained on this reach ~0.9 macro-F1 -- useful for seeing
how the tool behaves when the data actually separates.

Outputs (relative to repo root):
  examples/training_data/*.csv   -- 4 files, ~40k rows total, training format
  examples/large_dataset.csv     -- ~600 devices x several rows, input format

Run:  python scripts/generate_dataset.py [--seed N]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROTOCOLS = ["MQTT", "CoAP", "HTTPS", "LoRaWAN"]

# Device archetypes: (payload µ/σ bytes, throughput µ/σ Mbit/s, latency µ/σ ms,
#                     jitter µ/σ ms, packet_loss µ/σ %)
ARCHETYPES = {
    "smart_meter": ((28, 8), (0.4, 0.3), (180, 60), (2, 1.5), (3.0, 2.0)),
    "env_sensor":  ((70, 25), (4, 3), (25, 12), (3, 2), (1.2, 1.0)),
    "actuator":    ((110, 30), (8, 4), (18, 8), (6, 3), (0.8, 0.7)),
    "camera":      ((1600, 700), (120, 60), (60, 25), (4, 3), (0.6, 0.5)),
    "gateway":     ((900, 500), (60, 40), (45, 20), (8, 5), (1.0, 0.9)),
}


def _score(latency, jitter, loss, thr, payload) -> np.ndarray:
    """Return a [MQTT, CoAP, HTTPS, LoRaWAN] score vector for one measurement."""
    s = np.zeros(4)  # order matches PROTOCOLS
    # latency (ms)
    if latency < 30:
        s += [2, 2, 0, 0]
    elif latency < 100:
        s += [1, 2, 1, 0]
    else:
        s += [0, 1, 0, 2]
    # jitter (ms)
    if jitter < 5:
        s += [2, 0, 2, 0]
    elif jitter < 20:
        s += [1, 2, 0, 0]
    else:
        s += [0, 2, 0, 1]
    # packet loss (%)
    if loss < 1:
        s += [1, 0, 3, 0]
    elif loss < 5:
        s += [3, 2, 0, 0]
    else:
        s += [2, 0, 0, 2]
    # throughput (Mbit/s)
    if thr < 1:
        s += [0, 0, 0, 4]
    elif thr < 10:
        s += [2, 2, 0, 0]
    elif thr < 50:
        s += [1, 1, 1, 0]
    else:
        s += [0, 0, 3, 0]
    # payload (bytes)
    if payload < 50:
        s += [1, 0, 0, 2]
    elif payload < 256:
        s += [3, 2, 0, 0]
    elif payload < 1024:
        s += [0, 2, 1, 0]
    else:
        s += [0, 0, 4, 0]
    # LoRaWAN is physically unsuitable for big or fast traffic
    if payload > 512 or thr > 20:
        s[3] -= 6
    return s


def _draw(rng, archetype, n):
    (pm, ps), (tm, ts), (lm, ls), (jm, js), (km, ks) = ARCHETYPES[archetype]
    payload = np.clip(rng.normal(pm, ps, n), 8, None)
    thr = np.clip(rng.normal(tm, ts, n), 0.05, None)
    latency = np.clip(rng.normal(lm, ls, n), 1, None)
    jitter = np.clip(rng.normal(jm, js, n), 0.1, None)
    loss = np.clip(rng.normal(km, ks, n), 0, 40)
    return payload, thr, latency, jitter, loss


def _label(rng, payload, thr, latency, jitter, loss):
    y = []
    for p, t, la, ji, lo in zip(payload, thr, latency, jitter, loss):
        scores = _score(la, ji, lo, t, p) + rng.normal(0, 0.8, 4)  # noise
        y.append(PROTOCOLS[int(np.argmax(scores))])
    return np.array(y)


def build_training(rng) -> None:
    out_dir = REPO / "examples" / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()

    start = datetime(2025, 1, 1)
    for fi in range(4):
        rows = []
        for arche in ARCHETYPES:
            payload, thr, latency, jitter, loss = _draw(rng, arche, 2500)
            y = _label(rng, payload, thr, latency, jitter, loss)
            for k in range(len(y)):
                rows.append({
                    "timestamp": (start + timedelta(minutes=fi * 10000 + k)).isoformat(),
                    "payload_size": round(payload[k], 3),
                    "latency": round(latency[k], 3),
                    "jitter": round(jitter[k], 3),
                    "throughput": round(thr[k], 3),
                    "packet_loss": round(loss[k], 4),
                    "best_protocol": y[k],
                    "device_archetype": arche,
                })
        df = pd.DataFrame(rows).sample(frac=1, random_state=fi).reset_index(drop=True)
        df.to_csv(out_dir / f"synth_batch_{fi + 1}.csv", index=False)
    print(f"training_data: 4 files, {4 * len(ARCHETYPES) * 2500} rows")


def build_input(rng) -> None:
    rows = []
    n_devices = 600
    archetypes = list(ARCHETYPES)
    for d in range(n_devices):
        arche = archetypes[d % len(archetypes)]
        n = int(rng.integers(4, 12))
        payload, thr, latency, jitter, loss = _draw(rng, arche, n)
        y = _label(rng, payload, thr, latency, jitter, loss)
        # the device's "current" protocol: its usual best, but 40% are stale
        usual = pd.Series(y).value_counts().index[0]
        if rng.random() < 0.40:
            current = rng.choice([p for p in PROTOCOLS if p != usual])
        else:
            current = usual
        for k in range(n):
            rows.append({
                "device_id": f"{arche}-{d:03d}",
                "current_protocol": current,
                "payload_size": round(payload[k], 3),
                "latency": round(latency[k], 3),
                "jitter": round(jitter[k], 3),
                "throughput": round(thr[k], 3),
                "packet_loss": round(loss[k], 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "examples" / "large_dataset.csv", index=False)
    print(f"large_dataset.csv: {n_devices} devices, {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    build_training(np.random.default_rng(args.seed))
    build_input(np.random.default_rng(args.seed + 1))


if __name__ == "__main__":
    main()
