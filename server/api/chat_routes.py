from fastapi import APIRouter
from pydantic import BaseModel
from api.dataset_merger import process_and_merge_datasets

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    role: str
    patient_context: dict = {}

def _analyze_patient(patient_id: int) -> dict:
    df = process_and_merge_datasets()
    patient_df = df[df["Id"] == patient_id].fillna(0)
    if patient_df.empty:
        return {}
    latest = patient_df.iloc[-1]
    first  = patient_df.iloc[0]

    def avg(col, default=0):
        return round(patient_df[col].mean(), 1) if col in patient_df.columns else default

    return {
        "avg_steps":         round(avg("TotalSteps")),
        "avg_calories":      round(avg("Calories")),
        "avg_sleep":         round(avg("TotalMinutesAsleep")),
        "avg_hr":            avg("AverageHeartrate", 72),
        "avg_weight":        avg("WeightKg", 70),
        "avg_bmi":           avg("BMI", 22),
        "avg_very_active":   round(avg("VeryActiveMinutes")),
        "avg_sedentary":     round(avg("SedentaryMinutes")),
        "avg_mets":          avg("AvgMETs", 10),
        "sleep_quality":     avg("SleepQualityScore", 1),
        "steps_trend":       float(latest["TotalSteps"] - first["TotalSteps"]) if "TotalSteps" in latest.index else 0,
        "cal_trend":         float(latest["Calories"]   - first["Calories"])   if "Calories" in latest.index else 0,
        "days_tracked":      len(patient_df),
        "latest_weight":     float(latest["WeightKg"]) if "WeightKg" in latest.index else 70,
        "peak_hourly_steps": float(patient_df["PeakHourlySteps"].max()) if "PeakHourlySteps" in patient_df.columns else 0,
    }

def _doctor_response(msg: str, stats: dict) -> str:
    if not stats:
        return "No data available for this patient."
    msg = msg.lower()
    if any(w in msg for w in ["steps","activity","active","walking"]):
        trend = "improving" if stats["steps_trend"] > 0 else "declining"
        return (f"Patient averages {stats["avg_steps"]:,} steps/day over {stats["days_tracked"]} days. "
                f"Trend is {trend}. Very active minutes avg: {stats["avg_very_active"]} min/day. "
                f"{'Within healthy range.' if stats["avg_steps"] >= 7500 else "Below recommended 7,500 steps/day — consider intervention."}")
    elif any(w in msg for w in ["sleep","rest","insomnia"]):
        hrs = round(stats["avg_sleep"] / 60, 1)
        quality = "good" if stats["sleep_quality"] < 1.5 else "restless"
        return (f"Patient averages {hrs} hrs sleep/night. Sleep quality: {quality} (score: {stats["sleep_quality"]}, lower=better). "
                f"{'Adequate.' if hrs >= 7 else "Insufficient — consider sleep hygiene intervention."}")
    elif any(w in msg for w in ["heart","hr","pulse","cardio"]):
        status = "normal" if 60 <= stats["avg_hr"] <= 100 else ("elevated" if stats["avg_hr"] > 100 else "low")
        return f"Avg resting HR: {stats["avg_hr"]} bpm — {status}. {'No cardiovascular flags.' if status == "normal" else "Recommend cardiovascular evaluation."}"
    elif any(w in msg for w in ["weight","bmi","obese"]):
        bmi_status = "normal" if 18.5 <= stats["avg_bmi"] <= 24.9 else ("overweight" if stats["avg_bmi"] < 30 else "obese")
        return (f"Weight: {stats["latest_weight"]} kg. Avg BMI: {stats["avg_bmi"]} ({bmi_status}). "
                f"{'No weight concerns.' if bmi_status == "normal" else "Recommend dietary and activity intervention."}")
    elif any(w in msg for w in ["calories","diet","nutrition"]):
        return (f"Patient burns {stats["avg_calories"]:,} kcal/day avg. METs avg: {stats["avg_mets"]} "
                f"(>10 = active). Caloric trend {'▲' if stats["cal_trend"] > 0 else "▼"}.")
    elif any(w in msg for w in ["sedentary","inactive","sitting"]):
        concern = stats["avg_sedentary"] > 600
        return (f"Patient averages {stats["avg_sedentary"]} sedentary minutes/day ({round(stats["avg_sedentary"]/60,1)} hrs). "
                f"{'HIGH sedentary time — recommend movement breaks.' if concern else "Sedentary time within acceptable range."}")
    elif any(w in msg for w in ["summary","overview","report","overall"]):
        hrs = round(stats["avg_sleep"] / 60, 1)
        return (f"Patient Summary ({stats["days_tracked"]} days): Steps {stats["avg_steps"]:,}/day | "
                f"Calories {stats["avg_calories"]:,} kcal | Sleep {hrs} hrs | HR {stats["avg_hr"]} bpm | "
                f"BMI {stats["avg_bmi"]} | Sedentary {round(stats["avg_sedentary"]/60,1)} hrs/day | METs {stats["avg_mets"]}.")
    elif any(w in msg for w in ["risk","concern","danger"]):
        risks = []
        if stats["avg_steps"] < 5000:    risks.append("very low activity (<5k steps)")
        if stats["avg_sleep"] < 360:     risks.append("poor sleep (<6 hrs)")
        if stats["avg_hr"] > 100:        risks.append("elevated resting HR")
        if stats["avg_bmi"] > 30:        risks.append("obese BMI")
        if stats["avg_sedentary"] > 720: risks.append("excessive sedentary time (>12 hrs)")
        if stats["avg_mets"] < 10:       risks.append("low metabolic activity")
        return "No significant risks detected." if not risks else f"Risk factors: {", ".join(risks)}. Recommend follow-up."
    else:
        return f"Patient tracked for {stats["days_tracked"]} days. Ask about steps, sleep, heart rate, weight, BMI, sedentary time, calories, METs, or overall summary."

def _patient_response(msg: str, stats: dict) -> str:
    if not stats:
        return "I couldn't find your health data."
    msg = msg.lower()
    if any(w in msg for w in ["steps","walking","active"]):
        return (f"You are averaging {stats["avg_steps"]:,} steps/day. {'Great work!' if stats["avg_steps"] >= 7500 else "Try to reach 7,500–10,000 steps by adding short walks!"}"
                f" Your peak hourly steps recorded: {int(stats["peak_hourly_steps"])}.")
    elif any(w in msg for w in ["sleep","tired","rest"]):
        hrs = round(stats["avg_sleep"] / 60, 1)
        quality = "restful" if stats["sleep_quality"] < 1.5 else "restless"
        return f"You average {hrs} hrs of sleep and your sleep tends to be {quality}. {'Great!' if hrs >= 7 else "Try going to bed 30 min earlier each night."}"
    elif any(w in msg for w in ["heart","pulse","hr"]):
        return f"Your average heart rate is {stats["avg_hr"]} bpm. {'Normal range — great!' if 60 <= stats["avg_hr"] <= 100 else "Outside normal range — please see your doctor."}"
    elif any(w in msg for w in ["weight","bmi"]):
        bmi_status = "healthy" if 18.5 <= stats["avg_bmi"] <= 24.9 else "outside healthy range"
        return f"Your weight is around {stats["latest_weight"]} kg with a BMI of {stats["avg_bmi"]} ({bmi_status}). Talk to your doctor for personalized guidance."
    elif any(w in msg for w in ["calories","burn","energy"]):
        return f"You burn around {stats["avg_calories"]:,} calories/day. {'Keep it up!' if stats["avg_calories"] > 2000 else "Try moving more throughout the day!"}"
    elif any(w in msg for w in ["sedentary","sitting","inactive"]):
        hrs = round(stats["avg_sedentary"] / 60, 1)
        return f"You spend about {hrs} hours sitting/inactive per day. {'Try to break it up with short walks every hour!' if hrs > 8 else "Good job staying active!"}"
    elif any(w in msg for w in ["how am i","doing","health","summary","overall"]):
        hrs = round(stats["avg_sleep"] / 60, 1)
        return (f"Overall you are doing {'well' if stats["avg_steps"] >= 7500 and hrs >= 7 else "okay"}! "
                f"Steps: {stats["avg_steps"]:,}/day, Sleep: {hrs} hrs/night, HR: {stats["avg_hr"]} bpm, BMI: {stats["avg_bmi"]}.")
    else:
        return "Hi! Ask me about your steps, sleep, heart rate, weight, BMI, calories, sedentary time, or say 'how am I doing' for a full summary!"

@router.post("/chat")
def chat_with_bot(request: ChatRequest):
    patient_id = request.patient_context.get("patient_id")
    stats = _analyze_patient(patient_id) if patient_id else {}
    stats["patient_id"] = patient_id
    reply = _doctor_response(request.message, stats) if request.role == "doctor" else _patient_response(request.message, stats)
    return {"reply": reply, "role_used": request.role}