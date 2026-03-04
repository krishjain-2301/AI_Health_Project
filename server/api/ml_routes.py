from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from api.cnn_model import predict_health_trajectory
from api.tinyml_model import predict_tinyml

router = APIRouter()

class DayData(BaseModel):
    TotalSteps: float = 0
    Calories: float = 0
    TotalMinutesAsleep: float = 0
    AverageHeartrate: float = 72.0
    WeightKg: float = 70.0
    VeryActiveMinutes: float = 0
    SedentaryMinutes: float = 0
    DailyAvgIntensity: float = 0
    AvgMETs: float = 10.0
    SleepQualityScore: float = 1.0

class HealthRequest(BaseModel):
    patient_id: int
    recent_sequence: List[DayData]
    use_tinyml: bool = False

def _pad_sequence(seq_dicts: list) -> list:
    pad = {"TotalSteps":0,"Calories":0,"TotalMinutesAsleep":0,
           "AverageHeartrate":72,"WeightKg":70,"VeryActiveMinutes":0,
           "SedentaryMinutes":0,"DailyAvgIntensity":0,"AvgMETs":10,"SleepQualityScore":1}
    while len(seq_dicts) < 3:
        seq_dicts.insert(0, pad.copy())
    return seq_dicts[-3:]

@router.post("/predict_consequences")
def predict_health(data: HealthRequest):
    seq_dicts = [d.model_dump() for d in data.recent_sequence]
    seq_dicts = _pad_sequence(seq_dicts)
    try:
        result = predict_tinyml(seq_dicts) if data.use_tinyml else predict_health_trajectory(seq_dicts)
        consequences = []
        if isinstance(result, dict) and "risk_level" in result:
            label = result.get("model_type", "CNN Pattern Analysis")
            if result["risk_level"] == "High":
                consequences.append({
                    "factor": label, "risk": "High",
                    "description": f"High-risk trajectory detected ({result["confidence_pct"]}% confidence). Patient trending toward sedentary habits with cardiovascular and metabolic risk."
                })
            else:
                consequences.append({
                    "factor": label, "risk": "Low",
                    "description": f"Healthy trajectory detected ({result["confidence_pct"]}% confidence). Activity and caloric levels are adequate to minimize long-term risk."
                })
        else:
            consequences.append({"factor": "Model Error", "risk": "Unknown", "description": str(result)})
        return {"predictions": consequences}
    except Exception as e:
        return {"predictions": [{"factor": "Prediction Fault", "risk": "Unknown", "description": str(e)}]}