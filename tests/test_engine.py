from __future__ import annotations

import pandas as pd
import pytest

from protocol_advisor.engine import (
    FEATURES,
    PROTOCOLS,
    Engine,
    TrainingDataError,
)


def test_load_or_train_creates_and_reloads_model(tmp_path, training_dir):
    home = tmp_path / "home"
    eng = Engine(home=home)
    info = eng.load_or_train(str(training_dir / "*.csv"))

    assert (home / "model.pkl").exists()
    assert info.model_name
    assert 0.0 <= info.macro_f1 <= 1.0
    assert set(info.per_class_f1) == set(PROTOCOLS)
    assert len(info.training_files) == 3
    assert info.low_confidence_eval is False

    # A fresh Engine pointed at the same home loads the cache, no retrain.
    reloaded = Engine(home=home).load_or_train()
    assert reloaded.model_name == info.model_name
    assert reloaded.trained_at == info.trained_at


def test_predict_returns_valid_labels_and_normalised_probs(engine):
    frame = pd.DataFrame(
        [[80, 15, 4.0, 30, 0.02], [1500, 120, 3.0, 200, 0.01]], columns=FEATURES
    )
    preds = engine.predict(frame)

    assert len(preds) == 2
    for p in preds:
        assert p.recommended in PROTOCOLS
        assert set(p.probabilities) == set(PROTOCOLS)
        assert p.probabilities[p.recommended] == max(p.probabilities.values())
        assert abs(sum(p.probabilities.values()) - 1.0) < 0.02


def test_learns_the_synthetic_signal(engine):
    """Separable synthetic data -> the winning model should be well above chance."""
    assert engine.info.macro_f1 > 0.8


def test_corrupt_model_pkl_triggers_retrain(tmp_path, training_dir):
    home = tmp_path / "home"
    home.mkdir()
    (home / "model.pkl").write_bytes(b"not a real pickle")

    eng = Engine(home=home)
    info = eng.load_or_train(str(training_dir / "*.csv"))
    assert info.model_name  # did not raise; retrained instead


def test_no_training_data_raises(tmp_path):
    eng = Engine(home=tmp_path / "home")
    with pytest.raises(TrainingDataError):
        eng.retrain(str(tmp_path / "does-not-exist" / "*.csv"))


def test_tiny_dataset_sets_low_confidence_flag(tmp_path, training_dir):
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    files = sorted(training_dir.glob("*.csv"))
    subset = pd.read_csv(files[0]).groupby("best_protocol", group_keys=False).head(15)
    subset.to_csv(tiny / "only.csv", index=False)

    info = Engine(home=tmp_path / "home").retrain(str(tiny / "*.csv"))
    assert info.low_confidence_eval is True


def test_very_small_training_set_does_not_raise(tmp_path, training_dir):
    """4 classes x 3 rows: too few to stratify a 25% split -> plain split, no crash."""
    tiny = tmp_path / "vtiny"
    tiny.mkdir()
    files = sorted(training_dir.glob("*.csv"))
    pd.read_csv(files[0]).groupby("best_protocol", group_keys=False).head(3).to_csv(
        tiny / "only.csv", index=False
    )
    info = Engine(home=tmp_path / "home").retrain(str(tiny / "*.csv"))
    assert info.model_name


def test_predict_before_load_raises(tmp_path):
    eng = Engine(home=tmp_path / "home")
    with pytest.raises(RuntimeError):
        eng.predict(pd.DataFrame([[1, 2, 3, 4, 5]], columns=FEATURES))
