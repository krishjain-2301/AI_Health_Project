import numpy as np
import pandas as pd

from api import cnn_model
from api.cnn_model import (FEATURES, SEQUENCE_LENGTH, STEP_TARGET, ACTIVE_MINUTES_TARGET,
                           build_windows, check_anomalies, compute_metrics, is_low_activity,
                           patient_folds)


def _patient(pid, steps, very_active=0, fairly_active=0):
    n = len(steps)
    df = pd.DataFrame({f: np.ones(n) for f in FEATURES})
    df["Id"] = pid
    df["ActivityDate"] = pd.date_range("2016-04-12", periods=n)
    df["TotalSteps"] = steps
    df["VeryActiveMinutes"] = very_active
    df["FairlyActiveMinutes"] = fairly_active
    return df


def test_low_activity_needs_both_signals_low():
    assert is_low_activity(STEP_TARGET - 1, ACTIVE_MINUTES_TARGET - 1)
    assert not is_low_activity(STEP_TARGET, 0)                  # enough steps
    assert not is_low_activity(0, ACTIVE_MINUTES_TARGET)        # enough active minutes


def test_windows_label_the_next_day():
    df = _patient(1, [9000] * SEQUENCE_LENGTH + [2000, 9000])
    X, y, y_prev, groups = build_windows(df)
    assert X.shape == (2, SEQUENCE_LENGTH, len(FEATURES))
    assert y.tolist() == [1, 0]          # day 4 is low, day 5 is active
    assert y_prev.tolist() == [0, 1]     # label of the last input day
    assert groups.tolist() == [1, 1]


def test_days_not_worn_are_skipped():
    df = _patient(1, [9000] * SEQUENCE_LENGTH + [0, 2000])
    X, y, y_prev, _ = build_windows(df)
    assert y.tolist() == [1]             # the 0-step day is neither a target nor an input
    assert y_prev.tolist() == [0]
    assert (X[0, :, FEATURES.index("TotalSteps")] > 0).all()


def test_windows_never_span_two_patients():
    n = SEQUENCE_LENGTH
    df = pd.concat([_patient(1, [9000] * n), _patient(2, [9000] * n)])
    X, _, _, _ = build_windows(df)
    assert len(X) == 0                   # one window each, but no following day to forecast


def test_folds_are_by_patient_disjoint_and_deterministic():
    groups = np.repeat(np.arange(100, 133), 10)
    y = (groups % 3 == 0).astype(int)
    folds = patient_folds(groups, y)
    assert len(folds) == cnn_model.N_FOLDS
    for train_idx, test_idx in folds:
        assert not set(groups[train_idx]) & set(groups[test_idx])
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(y)))   # each day scored once
    again = patient_folds(groups, y)
    assert all((a[1] == b[1]).all() for a, b in zip(folds, again))


def test_metrics_report_the_at_risk_class():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2])
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == 0.5
    assert m["recall"] == m["per_class"]["Low-activity"]["recall"] == 0.5


def test_anomaly_check_ignores_placeholders():
    day = {"AverageHeartrate": 130, "HeartRateMeasured": False, "TotalSteps": 9000}
    assert check_anomalies([day]) == []
    day["HeartRateMeasured"] = True
    assert [a["field"] for a in check_anomalies([day])] == ["AverageHeartrate"]


def test_explanation_measures_change_against_typical_value(monkeypatch):
    baseline = np.full(len(FEATURES), 0.5, dtype=np.float32)
    monkeypatch.setattr(cnn_model, "get_dataset", lambda: {"feature_baseline": baseline})
    steps = FEATURES.index("TotalSteps")
    # A stand-in model whose risk falls as (scaled) steps rise
    predict = lambda batch: 1.0 - batch[:, :, steps].mean(axis=1)
    x = np.full((SEQUENCE_LENGTH, len(FEATURES)), 0.5, dtype=np.float32)
    x[:, steps] = 0.1
    seq = [{}] * SEQUENCE_LENGTH
    out = cnn_model.explain(predict, x, seq)
    assert out[0]["feature"] == "TotalSteps"
    assert out[0]["contribution_pct"] == 40.0        # low steps raise risk by 40 points
    assert all(e["contribution_pct"] == 0 for e in out[1:])
    assert not any(e["placeholder"] for e in out)   # no model input is a placeholder
