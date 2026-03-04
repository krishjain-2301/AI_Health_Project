"""
tinyml_model.py
Converts the trained Keras CNN to a TFLite FlatBuffer (TinyML format),
then uses the TFLite interpreter for fast, lightweight inference.
This simulates deploying the model to an edge device (e.g. microcontroller).
"""
import os
import numpy as np

TFLITE_PATH = os.path.join(os.path.dirname(__file__), 'tinyml_model.tflite')
SEQUENCE_LENGTH = 3
FEATURES = ['TotalSteps','Calories','TotalMinutesAsleep','AverageHeartrate','WeightKg']
SCALE    = np.array([10000.0, 3000.0, 480.0, 100.0, 100.0], dtype=np.float32)
_interpreter = None

def _convert_and_save(keras_model):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantization for TinyML
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"[tinyml] Converted and saved TFLite model ({len(tflite_model)//1024} KB)")
    return tflite_model

def get_interpreter():
    global _interpreter
    if _interpreter is not None:
        return _interpreter

    import tensorflow as tf

    if not os.path.exists(TFLITE_PATH):
        print("[tinyml] No TFLite model found, converting from CNN...")
        from api.cnn_model import get_model
        keras_model = get_model()
        if keras_model is None:
            print("[tinyml] Cannot convert — CNN model unavailable.")
            return None
        _convert_and_save(keras_model)

    _interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    _interpreter.allocate_tensors()
    print("[tinyml] TFLite interpreter ready.")
    return _interpreter

def predict_tinyml(recent_sequence: list) -> dict:
    """
    Run inference using the TFLite (TinyML) model.
    Same input format as cnn_model.predict_health_trajectory.
    """
    interpreter = get_interpreter()
    if interpreter is None:
        return {"error": "TinyML interpreter not available."}

    arr = [
        [s.get('TotalSteps',0), s.get('Calories',0), s.get('TotalMinutesAsleep',0),
         s.get('AverageHeartrate',72), s.get('WeightKg',70)]
        for s in recent_sequence
    ]
    input_arr = np.array([arr], dtype=np.float32) / SCALE

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    prediction = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    risk       = "Low" if prediction > 0.5 else "High"
    confidence = round(abs(prediction - 0.5) * 200, 2)

    return {
        "prediction":      prediction,
        "predicted_state": "Healthy" if prediction > 0.5 else "At-Risk",
        "risk_level":      risk,
        "confidence_pct":  confidence,
        "model_type":      "TinyML (TFLite quantized)"
    }