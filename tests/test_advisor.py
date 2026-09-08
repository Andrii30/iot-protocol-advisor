from __future__ import annotations

import pandas as pd
import pytest

from protocol_advisor.advisor import (
    KEEP,
    KEEP_LOW_CONFIDENCE,
    SWITCH,
    InputSchemaError,
    advise,
)


def test_one_row_per_device_with_expected_verdicts(engine, devices_frame):
    result = advise(devices_frame, engine)

    assert list(result["device_id"]) == ["dev-https", "dev-mqtt"]  # sorted
    by_id = result.set_index("device_id")

    assert by_id.loc["dev-mqtt", "recommended_protocol"] == "MQTT"
    assert by_id.loc["dev-mqtt", "verdict"] == KEEP

    assert by_id.loc["dev-https", "recommended_protocol"] == "HTTPS"
    assert by_id.loc["dev-https", "verdict"] == SWITCH


def test_n_samples_and_probabilities_present(engine, devices_frame):
    result = advise(devices_frame, engine)
    row = result.iloc[0]
    assert row["n_samples"] == 8
    assert isinstance(row["probabilities"], dict)
    assert abs(sum(row["probabilities"].values()) - 1.0) < 0.02


def test_high_threshold_forces_low_confidence_keep(engine, devices_frame):
    # threshold above any single-class probability -> never SWITCH
    result = advise(devices_frame, engine, switch_threshold=1.01)
    assert set(result["verdict"]) <= {KEEP, KEEP_LOW_CONFIDENCE}
    assert (result.loc[result["device_id"] == "dev-https", "verdict"] == KEEP_LOW_CONFIDENCE).all()


def test_unknown_current_protocol_still_gets_a_verdict(engine, devices_frame):
    devices_frame = devices_frame.copy()
    devices_frame["current_protocol"] = "SomethingElse"
    result = advise(devices_frame, engine)
    assert len(result) == 2
    assert set(result["verdict"]) <= {SWITCH, KEEP_LOW_CONFIDENCE}


def test_all_nan_rows_raise(engine, devices_frame):
    devices_frame = devices_frame.copy()
    devices_frame["jitter"] = float("nan")
    with pytest.raises(InputSchemaError):
        advise(devices_frame, engine)


def test_missing_current_protocol_column_raises(engine, devices_frame):
    with pytest.raises(InputSchemaError):
        advise(devices_frame.drop(columns=["current_protocol"]), engine)


def test_missing_feature_column_raises(engine, devices_frame):
    with pytest.raises(InputSchemaError):
        advise(devices_frame.drop(columns=["jitter"]), engine)
