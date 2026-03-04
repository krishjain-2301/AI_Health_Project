import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_cnn_model.keras")
SEQUENCE_LENGTH = 3

# Now using 10 features from all 18 CSV files
FEATURES = [
    "TotalSteps", "Calories", "TotalMinutesAsleep",
    "AverageHeartrate", "WeightKg", "VeryActiveMinutes",
    "SedentaryMinutes", "DailyAvgIntensity", "AvgMETs", "SleepQualityScore"
]
SCALE = np.array([10000.0, 3000.0, 480.0, 100.0, 100.0, 60.0, 1440.0, 1.0, 20.0, 3.0])
_model = None

def _build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    model = Sequential([
        Conv1D(64, kernel_size=2, activation="relu", input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),
        Dropout(0.3),
        Conv1D(32, kernel_size=1, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def _safe_get(df_group, col, default=0.0):
    if col in df_group.columns:
        return df_group[col].values
    return [default] * len(df_group)

def _prepare_data():
    from api.dataset_merger import process_and_merge_datasets
    df = process_and_merge_datasets()
    if df.empty:
        return np.array([]), np.array([])
    X_data, y_data = [], []
    for _, group in df.groupby("Id"):
        records = []
        for feat in FEATURES:
            if feat in group.columns:
                records.append(group[feat].fillna(0).values)
            else:
                records.append([0.0] * len(group))
        records = list(zip(*records))  # transpose to (n_days, n_features)
        records = list(records)
        for i in range(len(records) - SEQUENCE_LENGTH):
            X_data.append(records[i:i + SEQUENCE_LENGTH])
            y_data.append(1 if records[i + SEQUENCE_LENGTH][0] > 5000 else 0)
    if not X_data:
        return np.array([]), np.array([])
    return np.array(X_data, dtype=np.float32) / SCALE, np.array(y_data)

def get_model():
    global _model
    if _model is not None:
        return _model
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            # Check if saved model has same input shape (features may have changed)
            expected_shape = (None, SEQUENCE_LENGTH, len(FEATURES))
            if _model.input_shape != expected_shape:
                print(f"[cnn_model] Saved model input shape mismatch, retraining...")
                os.remove(MODEL_PATH)
            else:
                print("[cnn_model] Loaded saved CNN from disk.")
                return _model
        except Exception as e:
            print(f"[cnn_model] Saved model unusable, retraining: {e}")
    X, y = _prepare_data()
    if len(X) == 0:
        print("[cnn_model] No training data available.")
        return None
    _model = _build_model()
    _model.fit(X, y, epochs=25, batch_size=32, verbose=0, validation_split=0.1)
    _model.save(MODEL_PATH)
    print("[cnn_model] CNN trained and saved to disk.")
    return _model

def predict_health_trajectory(recent_sequence: list) -> dict:
    model = get_model()
    if model is None:
        return {"error": "Model not available."}
    arr = []
    for s in recent_sequence:
        arr.append([
            s.get("TotalSteps", 0),
            s.get("Calories", 0),
            s.get("TotalMinutesAsleep", 0),
            s.get("AverageHeartrate", 72),
            s.get("WeightKg", 70),
            s.get("VeryActiveMinutes", 0),
            s.get("SedentaryMinutes", 0),
            s.get("DailyAvgIntensity", 0),
            s.get("AvgMETs", 10),
            s.get("SleepQualityScore", 1),
        ])
    input_arr = np.array([arr], dtype=np.float32) / SCALE
    prediction = float(model.predict(input_arr, verbose=0)[0][0])
    risk = "Low" if prediction > 0.5 else "High"
    confidence = round(abs(prediction - 0.5) * 200, 2)
    return {
        "prediction": prediction,
        "predicted_state": "Healthy" if prediction > 0.5 else "At-Risk",
        "risk_level": risk,
        "confidence_pct": confidence
    }

try:
    get_model()
except Exception as e:
    print(f"[cnn_model] Startup error: {e}")