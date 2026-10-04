from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from api import cnn_model
from api.cnn_model import (predict_health_trajectory, FEATURES, PLACEHOLDER_FLAGS, SEQUENCE_LENGTH,
                           STEP_TARGET, ACTIVE_MINUTES_TARGET)
from api.dataset_merger import process_and_merge_datasets
from api.tinyml_model import predict_tinyml, get_tflite_report

router = APIRouter()

class DayData(BaseModel):
    TotalSteps: float = 0
    Calories: float = 0
    TotalMinutesAsleep: float = 0
    AverageHeartrate: float = 72.0
    WeightKg: float = 70.0
    VeryActiveMinutes: float = 0
    FairlyActiveMinutes: float = 0
    LightlyActiveMinutes: float = 0
    SedentaryMinutes: float = 0
    DailyAvgIntensity: float = 0
    AvgMETs: float = 10.0
    SleepQualityScore: float = 1.0

class HealthRequest(BaseModel):
    patient_id: int
    # Optional: when omitted the server uses the patient's most recent days
    recent_sequence: List[DayData] = []
    use_tinyml: bool = False

def _pad_sequence(seq_dicts: list) -> list:
    # Too few days: repeat the earliest one rather than inventing zero-step days
    pad = {k: v for k, v in seq_dicts[0].items() if k != "date"}
    while len(seq_dicts) < SEQUENCE_LENGTH:
        seq_dicts.insert(0, pad.copy())
    return seq_dicts[-SEQUENCE_LENGTH:]

def recent_sequence_for_patient(patient_id: int) -> list:
    """The patient's latest days as model input, with the measured/placeholder flags."""
    df = process_and_merge_datasets()
    patient_df = df[df["Id"] == patient_id].sort_values("ActivityDate")
    # Skip days the tracker was not worn (0 steps): they describe the device, not the person
    worn = patient_df[patient_df["TotalSteps"] > 0]
    patient_df = (worn if len(worn) else patient_df).tail(SEQUENCE_LENGTH)
    seq = []
    for _, row in patient_df.iterrows():
        # Heart rate and weight are not model inputs, but the range check reads them
        day = {f: float(row[f]) for f in list(FEATURES) + list(PLACEHOLDER_FLAGS)}
        for flag in PLACEHOLDER_FLAGS.values():
            day[flag] = bool(row[flag])
        day["date"] = str(row["ActivityDate"]).split(" ")[0]
        seq.append(day)
    return seq

def predict_for_patient(patient_id: int, use_tinyml: bool = False) -> dict:
    seq = recent_sequence_for_patient(patient_id)
    if not seq:
        return {}
    seq = _pad_sequence(seq)
    return predict_tinyml(seq) if use_tinyml else predict_health_trajectory(seq)

@router.post("/predict_consequences")
def predict_health(data: HealthRequest):
    if data.recent_sequence:
        seq_dicts = [d.model_dump() for d in data.recent_sequence]
    else:
        seq_dicts = recent_sequence_for_patient(data.patient_id)
        if not seq_dicts:
            raise HTTPException(status_code=404, detail=f"Patient {data.patient_id} not found")
    seq_dicts = _pad_sequence(seq_dicts)

    try:
        result = predict_tinyml(seq_dicts) if data.use_tinyml else predict_health_trajectory(seq_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    if "risk_level" not in result:
        raise HTTPException(status_code=503, detail=result.get("error", "Model not available."))
    result["based_on"] = [d["date"] for d in seq_dicts if d.get("date")]

    target = f"{STEP_TARGET:,} steps or {ACTIVE_MINUTES_TARGET} active minutes"
    if result["risk_level"] == "High":
        description = (f"The model expects tomorrow to be a low-activity day "
                       f"({result['risk_probability_pct']}% probability), falling short of {target}.")
    else:
        description = (f"The model expects tomorrow to meet the activity target of {target} "
                       f"({round(100 - result['risk_probability_pct'], 1)}% probability).")
    consequences = [{"factor": "Next-day activity forecast", "risk": result["risk_level"],
                     "description": description}]
    for a in result["anomalies"]:
        consequences.append({
            "factor": f"Out of range: {a['label']}",
            "risk": "High",
            "description": a["message"]
        })

    return {"predictions": consequences, "raw": result}

@router.get("/model_info")
def model_info():
    """How the model was built and how it scores on patients it never trained on."""
    info = cnn_model.get_evaluation()
    if not info:
        raise HTTPException(status_code=503, detail="Model not available.")
    return {**info, "tflite": get_tflite_report()}
