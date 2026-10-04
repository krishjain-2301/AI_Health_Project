"""
tinyml_model.py
Converts the trained Keras CNN to a TFLite FlatBuffer (TinyML format),
then uses the TFLite interpreter for fast, lightweight inference.
This simulates deploying the model to an edge device (e.g. microcontroller).
"""
import os
import threading
import numpy as np

from api import cnn_model
from api.cnn_model import SEQUENCE_LENGTH, FEATURES, MODEL_PATH

TFLITE_PATH = os.path.join(os.path.dirname(__file__), 'tinyml_model.tflite')
MODEL_TYPE  = "TinyML (TFLite quantized)"

_lock = threading.RLock()
_interpreter = None
_report = None

def _convert_and_save(keras_model):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantization for TinyML
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"[tinyml] Converted and saved TFLite model ({len(tflite_model)//1024} KB)")
    return tflite_model

def _is_stale() -> bool:
    """The .tflite must be rebuilt whenever the Keras model it came from changes."""
    if not os.path.exists(TFLITE_PATH):
        return True
    return os.path.exists(MODEL_PATH) and os.path.getmtime(TFLITE_PATH) < os.path.getmtime(MODEL_PATH)

def _load_interpreter():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

def get_interpreter():
    global _interpreter
    with _lock:
        if _interpreter is not None:
            return _interpreter

        keras_model = cnn_model.get_model()
        if keras_model is None:
            print("[tinyml] Cannot convert — CNN model unavailable.")
            return None

        if _is_stale():
            print("[tinyml] TFLite model missing or older than the CNN, converting...")
            _convert_and_save(keras_model)

        interpreter = _load_interpreter()
        expected = [1, SEQUENCE_LENGTH, len(FEATURES)]
        if list(interpreter.get_input_details()[0]['shape']) != expected:
            print("[tinyml] TFLite input shape does not match the CNN, reconverting...")
            _convert_and_save(keras_model)
            interpreter = _load_interpreter()

        _interpreter = interpreter
        print("[tinyml] TFLite interpreter ready.")
        return _interpreter

def tflite_predict_fn(batch: np.ndarray) -> np.ndarray:
    """P(next day is low-activity) for a batch of scaled windows, one invoke per row."""
    interpreter = get_interpreter()
    batch = np.asarray(batch, dtype=np.float32)
    out = np.zeros(len(batch), dtype=np.float32)
    with _lock:
        input_index  = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        for i, row in enumerate(batch):
            interpreter.set_tensor(input_index, row[None, :, :])
            interpreter.invoke()
            out[i] = interpreter.get_tensor(output_index)[0][0]
    return out

def predict_tinyml(recent_sequence: list) -> dict:
    """
    Run inference using the TFLite (TinyML) model.
    Same input format as cnn_model.predict_health_trajectory.
    """
    if get_interpreter() is None:
        return {"error": "TinyML interpreter not available."}
    return cnn_model.run_prediction(recent_sequence, tflite_predict_fn, MODEL_TYPE)

def get_tflite_report() -> dict:
    """Held-out metrics, size and speed of the TFLite model, plus how often it
    agrees with the Keras model it was converted from. Computed once per run."""
    global _report
    with _lock:
        if _report is not None:
            return _report
        if get_interpreter() is None:
            return {}
        data = cnn_model.get_dataset()
        report = cnn_model.evaluate(tflite_predict_fn)
        if not report:
            return {}
        keras_prob  = cnn_model.keras_predict_fn(data["X_test"])
        tflite_prob = tflite_predict_fn(data["X_test"])
        report["agreement_pct"] = round(float(((keras_prob >= 0.5) == (tflite_prob >= 0.5)).mean()) * 100, 1)
        report["max_probability_diff"] = round(float(np.abs(keras_prob - tflite_prob).max()), 4)
        report["size_kb"] = round(os.path.getsize(TFLITE_PATH) / 1024, 1)
        report["latency_ms"] = cnn_model.measure_latency_ms(tflite_predict_fn)
        _report = report
        return _report
