from __future__ import annotations

import pandas as pd

from protocol_advisor.advisor import advise
from protocol_advisor.baseline import rule_based_recommend


def _row(**kw):
    base = dict(payload_size=100, latency=30, jitter=4, throughput=10, packet_loss=1)
    base.update(kw)
    return pd.Series(base)


def test_rule_based_picks_expected_protocol():
    assert rule_based_recommend(_row(payload_size=5000)) == "HTTPS"
    assert rule_based_recommend(_row(throughput=80)) == "HTTPS"
    assert rule_based_recommend(_row(throughput=0.5, payload_size=30, latency=200)) == "LoRaWAN"
    assert rule_based_recommend(_row(packet_loss=9)) == "MQTT"
    assert rule_based_recommend(_row(jitter=2, latency=40, payload_size=200)) == "CoAP"


def test_advise_adds_rule_columns(engine, devices_frame):
    result = advise(devices_frame, engine)
    assert "rule_based" in result.columns
    assert "ml_agrees_rule" in result.columns
    assert result["ml_agrees_rule"].dtype == bool
    assert result["rule_based"].isin(["MQTT", "CoAP", "HTTPS", "LoRaWAN"]).all()
