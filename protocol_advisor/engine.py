"""Model training, selection, persistence and prediction.

Trains five scikit-learn classifiers on IoT network logs, evaluates them on a
stratified 25% holdout, and keeps the one with the best macro-F1. The fitted
model, scaler and label encoder are cached to ``model.pkl``.

Evaluation is a stratified 25% random split of the pooled rows. Three features
are derived from the raw five before training (see ``build_matrix``). See
README for what the reported numbers mean.
"""

from __future__ import annotations

import glob as _glob
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fixed order of the RAW input features (the public data contract).
FEATURES: list[str] = ["payload_size", "latency", "jitter", "throughput", "packet_loss"]

# Extra columns derived from the raw five before training / prediction.
DERIVED: list[str] = ["transfer_time", "burstiness", "log_payload"]


def build_matrix(x5: np.ndarray) -> np.ndarray:
    """Expand an (n, 5) array in FEATURES order into raw + derived features."""
    payload, latency, jitter, throughput, loss = (x5[:, i] for i in range(5))
    transfer_time = payload / np.clip(throughput, 0.01, None)
    burstiness = jitter / np.clip(latency, 1.0, None)
    log_payload = np.log1p(payload)
    return np.column_stack([x5, transfer_time, burstiness, log_payload])

# The 4 protocols this tool decides between, sorted (LabelEncoder order).
PROTOCOLS: list[str] = ["CoAP", "HTTPS", "LoRaWAN", "MQTT"]

LABEL_COLUMN = "best_protocol"
SCHEMA_VERSION = 2

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAINING_GLOB = str(_REPO_ROOT / "examples" / "training_data" / "*.csv")


class TrainingDataError(RuntimeError):
    """Raised when no usable training data can be found or loaded."""


@dataclass
class Prediction:
    recommended: str
    probabilities: dict[str, float]


@dataclass
class ModelInfo:
    model_name: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    trained_at: str
    training_files: list[str]
    n_rows: int
    sklearn_version: str = sklearn.__version__
    feature_importances: dict[str, float] | None = None
    low_confidence_eval: bool = False


def _model_zoo() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=42
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }


def _feature_importances(Xs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Impurity-based importances from a plain RandomForest fit on all rows.

    Computed independently of which model wins so the "why" shown in the UI is
    always available (the MLP, for instance, exposes no importances).
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(Xs, y)
    vals = np.asarray(rf.feature_importances_, dtype=float)
    total = vals.sum() or 1.0
    return {
        name: round(float(v / total), 4)
        for name, v in zip(FEATURES + DERIVED, vals)
    }


def _load_training_frame(training_glob: str) -> pd.DataFrame:
    paths = sorted(_glob.glob(training_glob))
    if not paths:
        raise TrainingDataError(f"No training CSV files match: {training_glob}")

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface a clean message
            raise TrainingDataError(f"Cannot read training file {path}: {exc}") from exc
        missing = [c for c in FEATURES + [LABEL_COLUMN] if c not in df.columns]
        if missing:
            raise TrainingDataError(
                f"Training file {path} is missing columns: {', '.join(missing)}"
            )
        df = df[FEATURES + [LABEL_COLUMN]].copy()
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=FEATURES + [LABEL_COLUMN])
    data = data[data[LABEL_COLUMN].isin(PROTOCOLS)]
    if data.empty:
        raise TrainingDataError(
            f"No valid rows (known protocol + all features) in: {training_glob}"
        )
    return data


class Engine:
    """Owns the trained model and answers protocol questions about it."""

    def __init__(self, home: Path | str | None = None):
        if home is None:
            home = os.environ.get("PROTOCOL_ADVISOR_HOME") or (
                Path.home() / ".iot-protocol-advisor"
            )
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.model_path = self.home / "model.pkl"

        self._model = None
        self._scaler: StandardScaler | None = None
        self._encoder: LabelEncoder | None = None
        self._info: ModelInfo | None = None

    # -- lifecycle ---------------------------------------------------------

    def load_or_train(self, training_glob: str | None = None) -> ModelInfo:
        if self.model_path.exists():
            try:
                self._load()
                return self.info
            except Exception:  # noqa: BLE001 - corrupt/old cache => retrain
                pass
        return self.retrain(training_glob or _DEFAULT_TRAINING_GLOB)

    def retrain(self, training_glob: str | None = None) -> ModelInfo:
        data = _load_training_frame(training_glob or _DEFAULT_TRAINING_GLOB)

        X = build_matrix(data[FEATURES].to_numpy(dtype=float))
        y_str = data[LABEL_COLUMN].to_numpy()
        source_files = sorted(data["source_file"].unique().tolist())

        encoder = LabelEncoder().fit(PROTOCOLS)
        y = encoder.transform(y_str)

        # Too little data for a trustworthy holdout number.
        class_counts = np.bincount(y, minlength=len(PROTOCOLS))
        low_confidence_eval = bool(len(y) < 200 or class_counts.min() < 10)

        labels = list(range(len(PROTOCOLS)))
        # Stratify when every present class has >= 2 rows and the test split can
        # still hold one row per class; otherwise fall back to a plain split.
        present = class_counts[class_counts > 0]
        stratify = y if (present.min() >= 2 and len(y) * 0.25 >= len(present)) else None
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=42, stratify=stratify
        )

        eval_scaler = StandardScaler().fit(X[tr_idx])
        Xtr = eval_scaler.transform(X[tr_idx])
        Xte = eval_scaler.transform(X[te_idx])

        best_name, best_f1, best_metrics = None, -1.0, {}
        for name, model in _model_zoo().items():
            model.fit(Xtr, y[tr_idx])
            pred = model.predict(Xte)
            macro = f1_score(y[te_idx], pred, average="macro", labels=labels, zero_division=0)
            if macro > best_f1:
                per_class = f1_score(
                    y[te_idx], pred, average=None, labels=labels, zero_division=0
                )
                best_name, best_f1 = name, macro
                best_metrics = {
                    "accuracy": float(accuracy_score(y[te_idx], pred)),
                    "macro_f1": float(macro),
                    "per_class_f1": {
                        proto: round(float(score), 4)
                        for proto, score in zip(encoder.classes_, per_class)
                    },
                }

        # Refit the winner on ALL rows for the shipped model.
        final_scaler = StandardScaler().fit(X)
        Xs_all = final_scaler.transform(X)
        final_model = _model_zoo()[best_name]
        final_model.fit(Xs_all, y)

        self._model = final_model
        self._scaler = final_scaler
        self._encoder = encoder
        self._info = ModelInfo(
            model_name=best_name,
            accuracy=round(best_metrics["accuracy"], 4),
            macro_f1=round(best_metrics["macro_f1"], 4),
            per_class_f1=best_metrics["per_class_f1"],
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            training_files=source_files,
            n_rows=len(y),
            feature_importances=_feature_importances(Xs_all, y),
            low_confidence_eval=low_confidence_eval,
        )
        self._save()
        return self.info

    # -- queries ---------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("Model not loaded; call load_or_train() first")
        missing = [c for c in FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f"Feature frame missing columns: {', '.join(missing)}")

        X = build_matrix(features[FEATURES].to_numpy(dtype=float))
        Xs = self._scaler.transform(X)
        probs = self._model.predict_proba(Xs)
        classes = self._encoder.inverse_transform(self._model.classes_)

        out: list[Prediction] = []
        for row in probs:
            dist = {proto: 0.0 for proto in PROTOCOLS}  # protocols absent from training -> 0
            dist.update({proto: round(float(p), 4) for proto, p in zip(classes, row)})
            recommended = max(dist, key=dist.get)
            out.append(Prediction(recommended=recommended, probabilities=dist))
        return out

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("Model not loaded; call load_or_train() first")
        return self._info

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        # Write to a temp file then rename, so a crash mid-write can't leave a
        # truncated model.pkl (the worker thread is a daemon and dies on exit).
        tmp = self.model_path.with_name(self.model_path.name + ".tmp")
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model": self._model,
                "scaler": self._scaler,
                "encoder": self._encoder,
                "info": self._info,
            },
            tmp,
        )
        os.replace(tmp, self.model_path)

    def _load(self) -> None:
        # Security note: joblib.load unpickles and can execute arbitrary code.
        # model.pkl is only ever written by this app into a user-owned dir
        # (self.home). Do not point PROTOCOL_ADVISOR_HOME at a directory whose
        # model.pkl came from an untrusted party; retrain instead.
        blob = joblib.load(self.model_path)
        if blob.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("model.pkl schema version mismatch")
        self._model = blob["model"]
        self._scaler = blob["scaler"]
        self._encoder = blob["encoder"]
        self._info = blob["info"]

        trained_with = getattr(self._info, "sklearn_version", "unknown")
        if trained_with != sklearn.__version__:
            warnings.warn(
                f"model.pkl was trained with scikit-learn {trained_with}, "
                f"running {sklearn.__version__}; retrain if predictions look off.",
                RuntimeWarning,
                stacklevel=2,
            )
