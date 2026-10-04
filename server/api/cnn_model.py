import os
import json
import time
import shutil
import threading
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_cnn_model.keras")
META_PATH  = os.path.join(os.path.dirname(__file__), "saved_cnn_model.meta.json")
# Models used only for scoring (one per cross-validation fold); rebuilt when missing
FOLD_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "cv_folds")
SEQUENCE_LENGTH = 7
SEED = 42
L2 = 0.03
EPOCHS = 100
N_FOLDS = 5
# Bump when the label definition or the training setup changes, so a model
# saved under the old definition is retrained instead of silently reused.
LABEL_VERSION = 3

# Heart rate and weight are left out: most people never recorded them, so they
# would mostly be placeholder values.
FEATURES = [
    "TotalSteps", "Calories", "TotalMinutesAsleep",
    "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes",
    "SedentaryMinutes", "DailyAvgIntensity", "AvgMETs", "SleepQualityScore"
]
SCALE = np.array([10000.0, 3000.0, 480.0, 60.0, 60.0, 300.0, 1440.0, 1.0, 20.0, 3.0], dtype=np.float32)

FEATURE_INFO = {
    "TotalSteps":         {"label": "Daily steps",         "unit": "steps"},
    "Calories":           {"label": "Calories burned",     "unit": "kcal"},
    "TotalMinutesAsleep": {"label": "Sleep",               "unit": "min"},
    "VeryActiveMinutes":  {"label": "Very active minutes", "unit": "min"},
    "FairlyActiveMinutes":  {"label": "Fairly active minutes",  "unit": "min"},
    "LightlyActiveMinutes": {"label": "Lightly active minutes", "unit": "min"},
    "SedentaryMinutes":   {"label": "Sedentary time",      "unit": "min"},
    "DailyAvgIntensity":  {"label": "Average intensity",   "unit": ""},
    "AvgMETs":            {"label": "METs",                "unit": "METs"},
    "SleepQualityScore":  {"label": "Sleep restlessness",  "unit": ""},
}
# Readings that are placeholders when the flag column is False
PLACEHOLDER_FLAGS = {"AverageHeartrate": "HeartRateMeasured", "WeightKg": "WeightMeasured"}

# ── Label ────────────────────────────────────────────────────────────────────
# The model forecasts whether the NEXT day will be a low-activity day. It is an
# activity forecast, not a clinical diagnosis.
STEP_TARGET = 7500
ACTIVE_MINUTES_TARGET = 30
CLASS_NAMES = ["Active", "Low-activity"]   # index = label value; 1 is the at-risk class
LABEL_DESCRIPTION = (
    f"A day is 'Low-activity' when it has fewer than {STEP_TARGET:,} steps and fewer than "
    f"{ACTIVE_MINUTES_TARGET} very/fairly active minutes. The model reads the previous "
    f"{SEQUENCE_LENGTH} days the tracker was worn and forecasts the next one. Days it was "
    "not worn (0 steps) are skipped."
)

THRESHOLDS = {
    "AverageHeartrate":  {"min": 60,   "max": 100,  "unit": "bpm",   "label": "Heart Rate"},
    "TotalSteps":        {"min": 3000, "max": 25000, "unit": "steps", "label": "Daily Steps"},
    "TotalMinutesAsleep":{"min": 300,  "max": 600,  "unit": "min",   "label": "Sleep"},
    "WeightKg":          {"min": 40,   "max": 150,  "unit": "kg",    "label": "Weight"},
    "SedentaryMinutes":  {"min": 0,    "max": 720,  "unit": "min",   "label": "Sedentary Time"},
    "Calories":          {"min": 1200, "max": 5000, "unit": "kcal",  "label": "Calories Burned"},
    "AvgMETs":           {"min": 8,    "max": 18,   "unit": "METs",  "label": "METs"},
    "VeryActiveMinutes": {"min": 0,    "max": 300,  "unit": "min",   "label": "Very Active Minutes"},
}

_lock = threading.RLock()
_model = None
_dataset = None
_fold_models = None
_evaluation = None


def is_low_activity(steps: float, active_minutes: float) -> bool:
    return steps < STEP_TARGET and active_minutes < ACTIVE_MINUTES_TARGET


def check_anomalies(recent_sequence: list) -> list:
    alerts = []
    if not recent_sequence:
        return alerts
    last = recent_sequence[-1]
    for field, bounds in THRESHOLDS.items():
        val = last.get(field, None)
        if val is None or val == 0:
            continue
        # A placeholder value says nothing about the patient
        flag = PLACEHOLDER_FLAGS.get(field)
        if flag and last.get(flag) is False:
            continue
        lo, hi = bounds["min"], bounds["max"]
        label, unit = bounds["label"], bounds["unit"]
        if val < lo:
            alerts.append({"field": field, "label": label, "value": val, "severity": "low",
                           "message": f"{label} is LOW at {val:.1f} {unit} (expected >= {lo})"})
        elif val > hi:
            alerts.append({"field": field, "label": label, "value": val, "severity": "high",
                           "message": f"{label} is HIGH at {val:.1f} {unit} (expected <= {hi})"})
    return alerts


# ── Data ─────────────────────────────────────────────────────────────────────
def build_windows(df):
    """Slide a SEQUENCE_LENGTH-day window over each patient's worn days.

    Returns X (scaled), y (1 = next day is low-activity), y_prev (label of the
    last input day, for the persistence baseline) and the patient id per window.
    """
    X, y, y_prev, groups = [], [], [], []
    for pid, group in df.groupby("Id"):
        group = group.sort_values("ActivityDate")
        # Days the tracker was not worn describe the device, not the person
        group = group[group["TotalSteps"].fillna(0) > 0]
        feats = np.stack([
            group[f].fillna(0).to_numpy(dtype=np.float32) if f in group.columns
            else np.zeros(len(group), dtype=np.float32)
            for f in FEATURES
        ], axis=1)
        steps  = group["TotalSteps"].fillna(0).to_numpy()
        active = (group["VeryActiveMinutes"].fillna(0) + group["FairlyActiveMinutes"].fillna(0)).to_numpy()
        for i in range(len(group) - SEQUENCE_LENGTH):
            t = i + SEQUENCE_LENGTH
            X.append(feats[i:t])
            y.append(int(is_low_activity(steps[t], active[t])))
            y_prev.append(int(is_low_activity(steps[t - 1], active[t - 1])))
            groups.append(pid)
    if not X:
        empty = np.array([])
        return empty, empty, empty, empty
    return (np.array(X, dtype=np.float32) / SCALE, np.array(y, dtype=np.int64),
            np.array(y_prev, dtype=np.int64), np.array(groups))


def patient_folds(groups, y, seed: int = SEED):
    """Deterministic cross-validation folds BY PATIENT: all of a person's days
    fall in one fold, so every day is scored by a model that never saw that person."""
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = min(N_FOLDS, len(set(np.asarray(groups).tolist())))
    if n_splits < 2:
        return []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def get_dataset():
    global _dataset
    with _lock:
        if _dataset is not None:
            return _dataset
        from api.dataset_merger import process_and_merge_datasets
        df = process_and_merge_datasets()
        X, y, y_prev, groups = build_windows(df) if not df.empty else (np.array([]),) * 4
        if len(X) == 0:
            _dataset = {"empty": True}
            return _dataset
        _dataset = {
            "empty": False,
            "X": X, "y": y, "y_prev": y_prev, "groups": groups,
            "folds": patient_folds(groups, y),
            # Typical (mean) value of each feature, in scaled units
            "feature_baseline": X.mean(axis=(0, 1)),
        }
        return _dataset


# ── Model ────────────────────────────────────────────────────────────────────
def _build_model(X_train):
    # Deliberately tiny and heavily regularised: with a few hundred windows from
    # ~20 people, a larger network memorises the training patients and scores
    # below the baselines on unseen ones.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Normalization, Conv1D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    norm = Normalization(axis=-1)
    norm.adapt(X_train)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, len(FEATURES))),
        norm,
        Conv1D(8, kernel_size=3, activation="relu", kernel_regularizer=l2(L2)),
        Flatten(),
        Dropout(0.3),
        Dense(1, activation="sigmoid", kernel_regularizer=l2(L2))
    ])
    model.compile(optimizer=Adam(3e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _expected_meta() -> dict:
    return {"label_version": LABEL_VERSION, "features": FEATURES,
            "sequence_length": SEQUENCE_LENGTH, "seed": SEED}


def _saved_model_is_current() -> bool:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return False
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    return all(meta.get(k) == v for k, v in _expected_meta().items())


def _fit(X, y):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    model = _build_model(X)
    # Low-activity days are the minority; without weighting the model mostly
    # predicts "Active" and misses them.
    p = float(y.mean())
    class_weight = {0: 0.5 / (1 - p), 1: 0.5 / p} if 0 < p < 1 else None
    model.fit(X, y, epochs=EPOCHS, batch_size=32, verbose=0, class_weight=class_weight)
    return model


def _train(data):
    print("\n[cnn_model] Training CNN model")
    print(f"[cnn_model]   {len(data['X'])} windows / {len(np.unique(data['groups']))} patients")
    model = _fit(data["X"], data["y"])
    model.save(MODEL_PATH)
    meta = _expected_meta()
    meta.update({"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "epochs_run": EPOCHS})
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # The scoring models belong to the previous setup
    shutil.rmtree(FOLD_DIR, ignore_errors=True)
    print(f"[cnn_model] CNN trained ({meta['epochs_run']} epochs) and saved to disk.")
    return model


def get_model():
    global _model
    with _lock:
        if _model is not None:
            return _model
        data = get_dataset()
        if data["empty"]:
            print("[cnn_model] No training data available.")
            return None
        import tensorflow as tf
        if _saved_model_is_current():
            try:
                _model = tf.keras.models.load_model(MODEL_PATH)
                if _model.input_shape != (None, SEQUENCE_LENGTH, len(FEATURES)):
                    print("[cnn_model] Saved model input shape mismatch, retraining...")
                    _model = None
                else:
                    print("[cnn_model] Loaded saved CNN from disk.")
            except Exception as e:
                print(f"[cnn_model] Saved model unusable, retraining: {e}")
                _model = None
        else:
            print("[cnn_model] No saved model for the current setup, training...")
        if _model is None:
            _model = _train(data)
        return _model


def _keras_fn(model):
    return lambda batch: model(np.asarray(batch, dtype=np.float32), training=False).numpy().reshape(-1)


def keras_predict_fn(batch: np.ndarray) -> np.ndarray:
    """P(next day is low-activity) for a batch of scaled windows."""
    return _keras_fn(get_model())(batch)


def get_fold_models() -> list:
    """One model per cross-validation fold, trained without that fold's patients.
    They are only used to score the approach; forecasts come from get_model()."""
    global _fold_models
    with _lock:
        if _fold_models is not None:
            return _fold_models
        data = get_dataset()
        if data["empty"] or get_model() is None:
            return []
        import tensorflow as tf
        paths = [os.path.join(FOLD_DIR, f"fold_{i}.keras") for i in range(len(data["folds"]))]
        models = None
        if all(os.path.exists(p) for p in paths):
            try:
                models = [tf.keras.models.load_model(p) for p in paths]
            except Exception as e:
                print(f"[cnn_model] Saved fold models unusable, retraining: {e}")
        if models is None:
            print(f"[cnn_model] Training {len(paths)} cross-validation models...")
            os.makedirs(FOLD_DIR, exist_ok=True)
            models = []
            for path, (train_idx, _) in zip(paths, data["folds"]):
                models.append(_fit(data["X"][train_idx], data["y"][train_idx]))
                models[-1].save(path)
        _fold_models = models
        return _fold_models


def out_of_fold(make_predict_fn) -> np.ndarray:
    """P(low-activity) for every window, each from the fold model that never saw
    its patient. make_predict_fn turns a fold's Keras model into a predict function."""
    data = get_dataset()
    prob = np.zeros(len(data["y"]), dtype=np.float32)
    for model, (_, test_idx) in zip(get_fold_models(), data["folds"]):
        prob[test_idx] = make_predict_fn(model)(data["X"][test_idx])
    return prob


def keras_out_of_fold() -> np.ndarray:
    return out_of_fold(_keras_fn)


# ── Evaluation ───────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 confusion_matrix, roc_auc_score)
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(precision_score(y_true, y_pred, pos_label=idx, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, pos_label=idx, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_pred, pos_label=idx, zero_division=0)), 4),
            "support":   int((y_true == idx).sum()),
        }
    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        # precision / recall / f1 are for the at-risk class ("Low-activity")
        "precision": per_class[CLASS_NAMES[1]]["precision"],
        "recall":    per_class[CLASS_NAMES[1]]["recall"],
        "f1":        per_class[CLASS_NAMES[1]]["f1"],
        # rows = actual, cols = predicted, in CLASS_NAMES order
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_class": per_class,
        "test_samples": int(len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return metrics


def measure_latency_ms(predict_fn, runs: int = 40) -> float:
    """Median single-sample inference time. Indicative only (this machine, warm)."""
    data = get_dataset()
    if data["empty"]:
        return 0.0
    sample = data["X"][:1]
    predict_fn(sample)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 3)


def get_evaluation() -> dict:
    """Cross-validated evaluation of the Keras CNN, computed once per server run."""
    global _evaluation
    with _lock:
        if _evaluation is not None:
            return _evaluation
        data = get_dataset()
        if data["empty"] or not data["folds"] or get_model() is None:
            return {}
        y = data["y"]
        majority = int(round(float(y.mean())))
        keras = compute_metrics(y, keras_out_of_fold())
        keras["size_kb"] = round(os.path.getsize(MODEL_PATH) / 1024, 1)
        keras["latency_ms"] = measure_latency_ms(keras_predict_fn)
        n_patients = int(len(np.unique(data["groups"])))
        _evaluation = {
            "label": {
                "description": LABEL_DESCRIPTION,
                "step_target": STEP_TARGET,
                "active_minutes_target": ACTIVE_MINUTES_TARGET,
                "class_names": CLASS_NAMES,
                "positive_class": CLASS_NAMES[1],
            },
            "features": [FEATURE_INFO[f]["label"] for f in FEATURES],
            "sequence_length": SEQUENCE_LENGTH,
            "split": {
                "method": "cross-validation by patient",
                "folds": len(data["folds"]),
                "patients": n_patients,
                "windows": int(len(y)),
                "low_activity_share": round(float(y.mean()), 4),
            },
            "baselines": {
                "majority":    compute_metrics(y, np.full(len(y), float(majority))),
                "persistence": compute_metrics(y, data["y_prev"].astype(float)),
            },
            "keras": keras,
        }
        print(f"[cnn_model] Cross-validated evaluation: accuracy={keras['accuracy']:.3f} "
              f"f1={keras['f1']:.3f} on {keras['test_samples']} windows "
              f"from {n_patients} patients, each scored by a model that never saw them.")
        return _evaluation


# ── Prediction + explanation ─────────────────────────────────────────────────
def _sequence_to_array(recent_sequence: list) -> np.ndarray:
    arr = [[
        s.get("TotalSteps", 0), s.get("Calories", 0),
        s.get("TotalMinutesAsleep", 0), s.get("VeryActiveMinutes", 0),
        s.get("FairlyActiveMinutes", 0), s.get("LightlyActiveMinutes", 0),
        s.get("SedentaryMinutes", 0), s.get("DailyAvgIntensity", 0),
        s.get("AvgMETs", 10), s.get("SleepQualityScore", 1),
    ] for s in recent_sequence]
    return np.array(arr, dtype=np.float32)


def explain(predict_fn, x_scaled: np.ndarray, recent_sequence: list) -> list:
    """Occlusion attribution: swap one feature at a time for its typical
    (training-mean) value and see how far the risk probability moves.

    contribution_pct > 0 means the patient's actual value RAISES the
    low-activity risk relative to a typical value; < 0 means it lowers it.
    """
    data = get_dataset()
    baseline = data["feature_baseline"]
    batch = np.repeat(x_scaled[None, :, :], len(FEATURES) + 1, axis=0)
    for i in range(len(FEATURES)):
        batch[i + 1, :, i] = baseline[i]
    probs = predict_fn(batch)
    raw_mean = (x_scaled * SCALE).mean(axis=0)
    typical = baseline * SCALE
    out = []
    for i, feat in enumerate(FEATURES):
        flag = PLACEHOLDER_FLAGS.get(feat)
        placeholder = bool(flag) and all(s.get(flag) is False for s in recent_sequence)
        out.append({
            "feature": feat,
            "label": FEATURE_INFO[feat]["label"],
            "unit": FEATURE_INFO[feat]["unit"],
            "contribution_pct": round(float(probs[0] - probs[i + 1]) * 100, 1),
            "patient_value": round(float(raw_mean[i]), 2),
            "typical_value": round(float(typical[i]), 2),
            "placeholder": placeholder,
        })
    out.sort(key=lambda d: abs(d["contribution_pct"]), reverse=True)
    return out


def run_prediction(recent_sequence: list, predict_fn, model_type: str) -> dict:
    x_scaled = _sequence_to_array(recent_sequence) / SCALE
    prob = float(predict_fn(x_scaled[None, :, :])[0])
    at_risk = prob >= 0.5
    return {
        "prediction": prob,                              # P(next day is low-activity)
        "risk_probability_pct": round(prob * 100, 1),
        "predicted_state": CLASS_NAMES[1] if at_risk else CLASS_NAMES[0],
        "risk_level": "High" if at_risk else "Low",
        "confidence_pct": round(max(prob, 1 - prob) * 100, 1),
        "anomalies": check_anomalies(recent_sequence),
        "explanation": explain(predict_fn, x_scaled, recent_sequence),
        "model_type": model_type,
    }


def predict_health_trajectory(recent_sequence: list) -> dict:
    if get_model() is None:
        return {"error": "Model not available."}
    return run_prediction(recent_sequence, keras_predict_fn, "CNN (Keras)")
