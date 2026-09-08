"""A transparent rule-based protocol selector, for comparison with the model.

These are hand-written thresholds over the five raw features — the kind of
decision table an operator might use without any ML. `advise()` runs this
alongside the model so you can see where they agree and where they differ.
"""

from __future__ import annotations

import pandas as pd


def rule_based_recommend(row: pd.Series) -> str:
    """Pick a protocol from one device's median feature values."""
    payload = row["payload_size"]
    latency = row["latency"]
    jitter = row["jitter"]
    throughput = row["throughput"]
    loss = row["packet_loss"]

    # Big or fast traffic -> HTTPS.
    if payload > 1024 or throughput > 50:
        return "HTTPS"
    # Tiny, slow, loss-tolerant links -> LoRaWAN.
    if throughput < 1.5 and payload < 120 and latency > 120:
        return "LoRaWAN"
    # Lossy channel -> MQTT with QoS.
    if loss > 5:
        return "MQTT"
    # Steady low-jitter medium links -> CoAP.
    if jitter < 8 and latency < 100 and payload < 1024:
        return "CoAP"
    # Default IoT pub/sub.
    return "MQTT"


def rule_based_column(devices_median: pd.DataFrame) -> pd.Series:
    return devices_median.apply(rule_based_recommend, axis=1)
