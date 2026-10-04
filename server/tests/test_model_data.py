import numpy as np
import pandas as pd

from api import cnn_model
from api.cnn_model import (FEATURES, SEQUENCE_LENGTH, STEP_TARGET, ACTIVE_MINUTES_TARGET,
                           build_windows, check_anomalies, compute_metrics, is_low_activity,
                           split_patients)


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
    df = _patient(1, [9000, 9000, 9000, 2000, 9000])
    X, y, y_prev, groups = build_windows(df)
    assert X.shape == (2, SEQUENCE_LENGTH, len(FEATURES))
    assert y.tolist() == [1, 0]          # day 4 is low, day 5 is active
    assert y_prev.tolist() == [0, 1]     # label of the last input day
    assert groups.tolist() == [1, 1]


def test_days_not_worn_are_not_targets():
    df = _patient(1, [9000, 9000, 9000, 0, 9000])
    _, y, _, _ = build_windows(df)
    assert y.tolist() == [0]             # the 0-step day is skipped as a target


def test_windows_never_span_two_patients():
    df = pd.concat([_patient(1, [9000] * 3), _patient(2, [9000] * 3)])
    X, _, _, _ = build_windows(df)
    assert len(X) == 0                   # 3 days each is not enough for a window + target


def test_split_is_by_patient_disjoint_and_deterministic():
    ids = list(range(100, 133))
    train, val, test = split_patients(ids)
    assert set(train) | set(val) | set(test) == set(ids)
    assert not (set(train) & set(test)) and not (set(train) & set(val)) and not (set(val) & set(test))
    assert len(test) == 7 and len(val) == 5
    assert split_patients(reversed(ids)) == (train, val, test)


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
    seq = [{"HeartRateMeasured": False, "WeightMeasured": True}] * SEQUENCE_LENGTH
    out = cnn_model.explain(predict, x, seq)
    assert out[0]["feature"] == "TotalSteps"
    assert out[0]["contribution_pct"] == 40.0        # low steps raise risk by 40 points
    assert all(e["contribution_pct"] == 0 for e in out[1:])
    by_name = {e["feature"]: e for e in out}
    assert by_name["AverageHeartrate"]["placeholder"] is True
    assert by_name["WeightKg"]["placeholder"] is False
