"""Synthetic, seeded fixtures. No network, no repo data files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protocol_advisor.engine import Engine

# Per-protocol feature centroids: [payload_size, latency, jitter, throughput, packet_loss]
_CENTROIDS = {
    "MQTT": [80, 15, 4.0, 30, 0.02],
    "CoAP": [60, 20, 1.0, 12, 0.05],
    "HTTPS": [1500, 120, 3.0, 200, 0.01],
    "LoRaWAN": [20, 200, 0.5, 1, 0.08],
}
_SCALE = [15, 5, 0.5, 5, 0.01]


def _make_frame(rows_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for proto, centre in _CENTROIDS.items():
        pts = rng.normal(centre, _SCALE, size=(rows_per_class, 5)).clip(min=0)
        df = pd.DataFrame(
            pts, columns=["payload_size", "latency", "jitter", "throughput", "packet_loss"]
        )
        df["best_protocol"] = proto
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def training_dir(tmp_path):
    """A directory of 3 training CSVs (distinct seeds => distinct 'source files')."""
    d = tmp_path / "training"
    d.mkdir()
    for i in range(3):
        _make_frame(rows_per_class=120, seed=100 + i).to_csv(
            d / f"logs_scenario_{i}.csv", index=False
        )
    return d


@pytest.fixture
def engine(tmp_path, training_dir):
    eng = Engine(home=tmp_path / "home")
    eng.load_or_train(str(training_dir / "*.csv"))
    return eng


@pytest.fixture
def devices_frame():
    """Two devices: one clearly MQTT-shaped, one clearly HTTPS-shaped."""
    rng = np.random.default_rng(7)
    mqtt = rng.normal(_CENTROIDS["MQTT"], _SCALE, size=(8, 5)).clip(min=0)
    https = rng.normal(_CENTROIDS["HTTPS"], _SCALE, size=(8, 5)).clip(min=0)
    cols = ["payload_size", "latency", "jitter", "throughput", "packet_loss"]

    d1 = pd.DataFrame(mqtt, columns=cols)
    d1["device_id"] = "dev-mqtt"
    d1["current_protocol"] = "MQTT"  # matches its shape -> KEEP

    d2 = pd.DataFrame(https, columns=cols)
    d2["device_id"] = "dev-https"
    d2["current_protocol"] = "CoAP"  # mismatched -> SWITCH

    return pd.concat([d1, d2], ignore_index=True)
