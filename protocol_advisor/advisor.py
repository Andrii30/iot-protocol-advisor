"""Turn per-measurement rows into one keep/switch verdict per device."""

from __future__ import annotations

import pandas as pd

from protocol_advisor.baseline import rule_based_recommend
from protocol_advisor.engine import FEATURES, Engine

REQUIRED_COLUMNS: list[str] = ["device_id", "current_protocol", *FEATURES]

KEEP = "KEEP"
SWITCH = "SWITCH"
KEEP_LOW_CONFIDENCE = "KEEP_LOW_CONFIDENCE"

DEFAULT_SWITCH_THRESHOLD = 0.55


class InputSchemaError(ValueError):
    """Raised when the input frame is missing required columns."""


def advise(
    devices: pd.DataFrame,
    engine: Engine,
    switch_threshold: float = DEFAULT_SWITCH_THRESHOLD,
) -> pd.DataFrame:
    """One row per device: current vs recommended protocol and a verdict.

    Columns returned: device_id, current_protocol, recommended_protocol,
    confidence, verdict, n_samples, factor_1, factor_2, rule_based,
    ml_agrees_rule, probabilities.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in devices.columns]
    if missing:
        raise InputSchemaError(
            "Input is missing required columns: " + ", ".join(missing)
        )

    devices = devices.dropna(subset=REQUIRED_COLUMNS)
    if devices.empty:
        raise InputSchemaError("Input has no rows with all required columns present.")

    grouped = devices.groupby("device_id", sort=True)
    agg = grouped[FEATURES].median()
    agg["current_protocol"] = grouped["current_protocol"].agg(
        lambda s: s.value_counts().index[0]
    )
    agg["n_samples"] = grouped.size()
    agg = agg.reset_index()

    predictions = engine.predict(agg[FEATURES])

    importances = engine.info.feature_importances or {}
    raw = {k: v for k, v in importances.items() if k in FEATURES}
    top_features = (sorted(raw, key=raw.get, reverse=True)[:2] + [None, None])[:2]

    rows = []
    for (_, dev), pred in zip(agg.iterrows(), predictions):
        current = dev["current_protocol"]
        recommended = pred.recommended
        confidence = pred.probabilities[recommended]

        if current == recommended:
            verdict = KEEP
        elif confidence >= switch_threshold:
            verdict = SWITCH
        else:
            verdict = KEEP_LOW_CONFIDENCE

        factors = [
            f"{name}={dev[name]:.3g}" if name is not None else "" for name in top_features
        ]
        rule = rule_based_recommend(dev)

        rows.append(
            {
                "device_id": dev["device_id"],
                "current_protocol": current,
                "recommended_protocol": recommended,
                "confidence": round(float(confidence), 4),
                "verdict": verdict,
                "n_samples": int(dev["n_samples"]),
                "factor_1": factors[0],
                "factor_2": factors[1],
                "rule_based": rule,
                "ml_agrees_rule": bool(rule == recommended),
                "probabilities": pred.probabilities,
            }
        )

    return pd.DataFrame(rows)
