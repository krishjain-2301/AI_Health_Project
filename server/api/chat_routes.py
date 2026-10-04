import os
import json
import threading
from typing import List, Literal

from fastapi import APIRouter
from pydantic import BaseModel
from api.dataset_merger import process_and_merge_datasets

router = APIRouter()

CHAT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
# Gemini models to try, in order. Individual models are often briefly
# overloaded (503), so a busy one is skipped in favour of the next.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODEL", "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3-flash-preview").split(",") if m.strip()]
# Errors that mean "this model can't answer right now", not "the request is wrong"
GEMINI_SKIP_CODES = (404, 429, 500, 503)
MAX_HISTORY_TURNS = 40

_client = None
_gemini_client = None
_client_lock = threading.Lock()

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    role: str
    patient_context: dict = {}
    history: List[ChatTurn] = []

def _analyze_patient(patient_id: int) -> dict:
    df = process_and_merge_datasets()
    patient_df = df[df["Id"] == patient_id]
    if patient_df.empty:
        return {}
    latest = patient_df.iloc[-1]
    first  = patient_df.iloc[0]

    # Sleep, heart rate and weight are only averaged over days they were really
    # recorded; None means "never recorded", not zero.
    def avg(col, mask=None, digits=1):
        rows = patient_df if mask is None else patient_df[patient_df[mask]]
        return round(float(rows[col].mean()), digits) if len(rows) else None

    weight_rows = patient_df[patient_df["WeightMeasured"]]
    return {
        "days_tracked":        len(patient_df),
        "avg_steps":           round(avg("TotalSteps")),
        "avg_calories":        round(avg("Calories")),
        "avg_very_active":     round(avg("VeryActiveMinutes")),
        "avg_sedentary":       round(avg("SedentaryMinutes")),
        "avg_mets":            avg("AvgMETs"),
        "steps_trend":         float(latest["TotalSteps"] - first["TotalSteps"]),
        "cal_trend":           float(latest["Calories"] - first["Calories"]),
        "peak_hourly_steps":   float(patient_df["PeakHourlySteps"].max()),
        "sleep_nights_logged": int(patient_df["SleepLogged"].sum()),
        "avg_sleep":           avg("TotalMinutesAsleep", "SleepLogged", 0),
        "sleep_quality":       avg("SleepQualityScore", "SleepLogged", 2),
        "hr_days_measured":    int(patient_df["HeartRateMeasured"].sum()),
        "avg_hr":              avg("AverageHeartrate", "HeartRateMeasured"),
        "weight_days_measured": len(weight_rows),
        "latest_weight":       round(float(weight_rows.iloc[-1]["WeightKg"]), 1) if len(weight_rows) else None,
        "avg_bmi":             avg("BMI", "WeightMeasured"),
    }

# ── Rule-based fallback (used when no LLM API key is configured) ──────────────
def _doctor_response(msg: str, stats: dict) -> str:
    if not stats:
        return "No data available for this patient."
    msg = msg.lower()
    if any(w in msg for w in ["steps","activity","active","walking"]):
        trend = "improving" if stats["steps_trend"] > 0 else "declining"
        verdict = ("Within healthy range." if stats["avg_steps"] >= 7500
                   else "Below recommended 7,500 steps/day — consider intervention.")
        return (f"Patient averages {stats['avg_steps']:,} steps/day over {stats['days_tracked']} days. "
                f"Trend is {trend}. Very active minutes avg: {stats['avg_very_active']} min/day. {verdict}")
    elif any(w in msg for w in ["sleep","rest","insomnia"]):
        if stats["avg_sleep"] is None:
            return "No sleep was logged for this patient, so sleep cannot be assessed."
        hrs = round(stats["avg_sleep"] / 60, 1)
        quality = "good" if stats["sleep_quality"] < 1.5 else "restless"
        verdict = "Adequate." if hrs >= 7 else "Insufficient — consider sleep hygiene intervention."
        return (f"Patient averages {hrs} hrs sleep/night over {stats['sleep_nights_logged']} logged nights. "
                f"Sleep quality: {quality} (score: {stats['sleep_quality']}, lower=better). {verdict}")
    elif any(w in msg for w in ["heart","hr","pulse","cardio"]):
        if stats["avg_hr"] is None:
            return "No heart-rate data was recorded for this patient."
        status = "normal" if 60 <= stats["avg_hr"] <= 100 else ("elevated" if stats["avg_hr"] > 100 else "low")
        verdict = "No cardiovascular flags." if status == "normal" else "Recommend cardiovascular evaluation."
        return (f"Avg daily HR: {stats['avg_hr']} bpm over {stats['hr_days_measured']} measured days — "
                f"{status}. {verdict}")
    elif any(w in msg for w in ["weight","bmi","obese"]):
        if stats["latest_weight"] is None:
            return "No weight or BMI was logged for this patient."
        bmi = stats["avg_bmi"]
        bmi_status = "normal" if 18.5 <= bmi <= 24.9 else ("underweight" if bmi < 18.5 else ("overweight" if bmi < 30 else "obese"))
        verdict = "No weight concerns." if bmi_status == "normal" else "Recommend dietary and activity review."
        return f"Weight: {stats['latest_weight']} kg. Avg BMI: {bmi} ({bmi_status}). {verdict}"
    elif any(w in msg for w in ["calories","diet","nutrition"]):
        arrow = "▲" if stats["cal_trend"] > 0 else "▼"
        return (f"Patient burns {stats['avg_calories']:,} kcal/day avg. METs avg: {stats['avg_mets']} "
                f"(>10 = active). Caloric trend {arrow}.")
    elif any(w in msg for w in ["sedentary","inactive","sitting"]):
        hrs = round(stats["avg_sedentary"] / 60, 1)
        verdict = ("HIGH sedentary time — recommend movement breaks." if stats["avg_sedentary"] > 600
                   else "Sedentary time within acceptable range.")
        return f"Patient averages {stats['avg_sedentary']} sedentary minutes/day ({hrs} hrs). {verdict}"
    elif any(w in msg for w in ["summary","overview","report","overall"]):
        sleep = f"{round(stats['avg_sleep'] / 60, 1)} hrs" if stats["avg_sleep"] is not None else "not logged"
        hr    = f"{stats['avg_hr']} bpm" if stats["avg_hr"] is not None else "not recorded"
        bmi   = stats["avg_bmi"] if stats["avg_bmi"] is not None else "not logged"
        return (f"Patient Summary ({stats['days_tracked']} days): Steps {stats['avg_steps']:,}/day | "
                f"Calories {stats['avg_calories']:,} kcal | Sleep {sleep} | HR {hr} | "
                f"BMI {bmi} | Sedentary {round(stats['avg_sedentary'] / 60, 1)} hrs/day | METs {stats['avg_mets']}.")
    elif any(w in msg for w in ["risk","concern","danger"]):
        risks = []
        if stats["avg_steps"] < 5000:    risks.append("very low activity (<5k steps)")
        if stats["avg_sleep"] is not None and stats["avg_sleep"] < 360: risks.append("poor sleep (<6 hrs)")
        if stats["avg_hr"] is not None and stats["avg_hr"] > 100:       risks.append("elevated average HR")
        if stats["avg_bmi"] is not None and stats["avg_bmi"] > 30:      risks.append("obese BMI")
        if stats["avg_sedentary"] > 720: risks.append("excessive sedentary time (>12 hrs)")
        if stats["avg_mets"] < 10:       risks.append("low metabolic activity")
        return "No significant risks detected." if not risks else f"Risk factors: {', '.join(risks)}. Recommend follow-up."
    else:
        return (f"Patient tracked for {stats['days_tracked']} days. Ask about steps, sleep, heart rate, "
                "weight, BMI, sedentary time, calories, METs, or overall summary.")

def _patient_response(msg: str, stats: dict) -> str:
    if not stats:
        return "I couldn't find your health data."
    msg = msg.lower()
    if any(w in msg for w in ["steps","walking","active"]):
        cheer = "Great work!" if stats["avg_steps"] >= 7500 else "Try to reach 7,500–10,000 steps by adding short walks!"
        return (f"You are averaging {stats['avg_steps']:,} steps/day. {cheer}"
                f" Your peak hourly steps recorded: {int(stats['peak_hourly_steps'])}.")
    elif any(w in msg for w in ["sleep","tired","rest"]):
        if stats["avg_sleep"] is None:
            return "Your tracker didn't log any sleep, so I can't tell you how you've been sleeping."
        hrs = round(stats["avg_sleep"] / 60, 1)
        quality = "restful" if stats["sleep_quality"] < 1.5 else "restless"
        cheer = "Great!" if hrs >= 7 else "Try going to bed 30 min earlier each night."
        return f"You average {hrs} hrs of sleep and your sleep tends to be {quality}. {cheer}"
    elif any(w in msg for w in ["heart","pulse","hr"]):
        if stats["avg_hr"] is None:
            return "Your tracker didn't record heart rate, so I don't have that for you."
        verdict = "Normal range — great!" if 60 <= stats["avg_hr"] <= 100 else "Outside normal range — please see your doctor."
        return f"Your average heart rate is {stats['avg_hr']} bpm. {verdict}"
    elif any(w in msg for w in ["weight","bmi"]):
        if stats["latest_weight"] is None:
            return "No weight was logged, so I can't comment on weight or BMI."
        bmi_status = "healthy" if 18.5 <= stats["avg_bmi"] <= 24.9 else "outside healthy range"
        return (f"Your weight is around {stats['latest_weight']} kg with a BMI of {stats['avg_bmi']} "
                f"({bmi_status}). Talk to your doctor for personalized guidance.")
    elif any(w in msg for w in ["calories","burn","energy"]):
        cheer = "Keep it up!" if stats["avg_calories"] > 2000 else "Try moving more throughout the day!"
        return f"You burn around {stats['avg_calories']:,} calories/day. {cheer}"
    elif any(w in msg for w in ["sedentary","sitting","inactive"]):
        hrs = round(stats["avg_sedentary"] / 60, 1)
        cheer = "Try to break it up with short walks every hour!" if hrs > 8 else "Good job staying active!"
        return f"You spend about {hrs} hours sitting/inactive per day. {cheer}"
    elif any(w in msg for w in ["how am i","doing","health","summary","overall"]):
        parts = [f"Steps: {stats['avg_steps']:,}/day"]
        if stats["avg_sleep"] is not None: parts.append(f"Sleep: {round(stats['avg_sleep'] / 60, 1)} hrs/night")
        if stats["avg_hr"] is not None:    parts.append(f"HR: {stats['avg_hr']} bpm")
        if stats["avg_bmi"] is not None:   parts.append(f"BMI: {stats['avg_bmi']}")
        well = stats["avg_steps"] >= 7500 and (stats["avg_sleep"] is None or stats["avg_sleep"] >= 420)
        return f"Overall you are doing {'well' if well else 'okay'}! {', '.join(parts)}."
    else:
        return ("Hi! Ask me about your steps, sleep, heart rate, weight, BMI, calories, sedentary time, "
                "or say 'how am I doing' for a full summary!")

def _rule_reply(message: str, role: str, stats: dict) -> str:
    return _doctor_response(message, stats) if role == "doctor" else _patient_response(message, stats)

# ── LLM providers ────────────────────────────────────────────────────────────
def _gemini_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def llm_provider():
    """Which LLM answers the chat: "gemini", "claude", or None for the built-in rules."""
    if _gemini_key():
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    return None

def _get_gemini_client():
    global _gemini_client
    with _client_lock:
        if _gemini_client is None:
            from google import genai
            _gemini_client = genai.Client(api_key=_gemini_key())
        return _gemini_client

def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()
        return _client

AUDIENCE = {
    "doctor": ("You are talking to a clinician reviewing this patient's record. Be concise and "
               "clinical: lead with the numbers, call out trends, and say what would merit follow-up."),
    "patient": ("You are talking to the patient themself. Be warm and plain-spoken, avoid jargon, "
                "and suggest small, practical next steps. For symptoms, medication or diagnosis "
                "questions, point them to their own clinician."),
}

SYSTEM_TEMPLATE = """You are the assistant inside HealthAI, a dashboard showing one person's Fitbit data \
(steps, calories, activity minutes, sleep, heart rate, weight) alongside a next-day activity forecast.

{audience}

Answer from the patient data below. It is everything the dashboard knows about this person, so if a \
question needs something that is not there, say so rather than guessing. A null value means that metric \
was never recorded for this person; say it is missing instead of treating it as zero or as normal.

The forecast is an activity forecast: the probability that tomorrow is a low-activity day (under \
{step_target:,} steps and under {active_target} active minutes), with the factors that moved it. It is not \
a diagnosis and says nothing about disease risk, so describe it in those terms. This is a demo built on a \
public research dataset, not a medical device.

The chat panel is narrow and shows plain text only, so write a few short sentences with no Markdown \
(no headings, bullet lists, bold or tables).

<patient_data>
{patient_data}
</patient_data>"""

def _patient_data(patient_id, stats: dict) -> str:
    df = process_and_merge_datasets()
    recent = df[df["Id"] == patient_id].sort_values("ActivityDate").tail(7)
    days = []
    for _, row in recent.iterrows():
        days.append({
            "date": str(row["ActivityDate"]).split(" ")[0],
            "steps": int(row["TotalSteps"]),
            "calories": int(row["Calories"]),
            "very_active_min": int(row["VeryActiveMinutes"]),
            "fairly_active_min": int(row["FairlyActiveMinutes"]),
            "sedentary_min": int(row["SedentaryMinutes"]),
            "sleep_min": int(row["TotalMinutesAsleep"]) if row["SleepLogged"] else None,
            "avg_heart_rate": round(float(row["AverageHeartrate"]), 1) if row["HeartRateMeasured"] else None,
        })
    data = {"patient_id": patient_id, "averages_over_tracking_period": stats, "last_7_days": days}

    try:
        from api.ml_routes import predict_for_patient
        result = predict_for_patient(patient_id)
        if "risk_level" in result:
            data["next_day_forecast"] = {
                "predicted_state": result["predicted_state"],
                "low_activity_probability_pct": result["risk_probability_pct"],
                "top_factors": [
                    {"factor": e["label"],
                     "effect_on_risk_pct_points": e["contribution_pct"],
                     "patient_3_day_avg": None if e["placeholder"] else e["patient_value"],
                     "typical_value": e["typical_value"], "unit": e["unit"]}
                    for e in result["explanation"][:4]
                ],
                "out_of_range_readings": [a["message"] for a in result["anomalies"]],
            }
    except Exception as e:
        print(f"[chat] Forecast unavailable for chat context: {e}")

    return json.dumps(data, indent=2)

def _build_prompt(request: ChatRequest, patient_id, stats: dict):
    """System prompt and message list (prior turns + the new question)."""
    from api.cnn_model import STEP_TARGET, ACTIVE_MINUTES_TARGET
    role = request.role if request.role in AUDIENCE else "patient"
    system = SYSTEM_TEMPLATE.format(
        audience=AUDIENCE[role], step_target=STEP_TARGET, active_target=ACTIVE_MINUTES_TARGET,
        patient_data=_patient_data(patient_id, stats) if stats else "No patient is selected.",
    )

    history = [{"role": t.role, "content": t.content} for t in request.history[-MAX_HISTORY_TURNS:]]
    while history and history[0]["role"] != "user":
        history.pop(0)
    return system, history + [{"role": "user", "content": request.message}]

EMPTY_REPLY = "I didn't get an answer back. Please try asking again."

def _gemini_generate(model: str, system: str, messages: list) -> str:
    from google.genai import types
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    response = _get_gemini_client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return (response.text or "").strip() or EMPTY_REPLY

def _gemini_reply(system: str, messages: list) -> str:
    """Ask each configured model once, in order, until one answers. One request
    per model and no retries, to stay inside the free-tier rate limit."""
    from google.genai import errors
    for i, model in enumerate(GEMINI_MODELS):
        try:
            return _gemini_generate(model, system, messages)
        except errors.APIError as e:
            last = i == len(GEMINI_MODELS) - 1
            if e.code not in GEMINI_SKIP_CODES or last:
                raise
            print(f"[chat] {model} unavailable ({e.code}), trying {GEMINI_MODELS[i + 1]}")

def _claude_reply(system: str, messages: list) -> str:
    response = _get_client().with_options(timeout=90.0).beta.messages.create(
        model=CHAT_MODEL,
        max_tokens=16000,
        # If a safety classifier declines the request, re-run it on Anthropic's
        # recommended fallback model instead of returning the refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=system,
        messages=messages,
    )
    if response.stop_reason == "refusal":
        return "I'm not able to help with that request. Try asking about this patient's activity, sleep or forecast."
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or EMPTY_REPLY

@router.get("/chat/status")
def chat_status():
    provider = llm_provider()
    model = {"gemini": GEMINI_MODELS[0], "claude": CHAT_MODEL}.get(provider)
    return {"llm_enabled": provider is not None, "provider": provider, "model": model}

@router.post("/chat")
def chat_with_bot(request: ChatRequest):
    patient_id = request.patient_context.get("patient_id")
    stats = _analyze_patient(patient_id) if patient_id else {}

    def rules(notice=None):
        return {"reply": _rule_reply(request.message, request.role, stats),
                "role_used": request.role, "source": "rules", "notice": notice}

    provider = llm_provider()
    if provider is None:
        return rules()

    system, messages = _build_prompt(request, patient_id, stats)

    if provider == "gemini":
        from google.genai import errors
        try:
            reply = _gemini_reply(system, messages)
        except errors.APIError as e:
            print(f"[chat] Gemini API error {e.code}: {e.message}")
            if e.code in (400, 401, 403, 404):
                return rules(f"Gemini rejected the request ({e.code}), so this is a built-in answer. "
                             "Check GEMINI_API_KEY and GEMINI_MODEL in server/.env.")
            if e.code == 429:
                return rules("Gemini's rate limit or quota was reached, so this is a built-in answer. Try again shortly.")
            return rules(f"Gemini returned an error ({e.code}), so this is a built-in answer.")
        except Exception as e:
            print(f"[chat] Gemini call failed: {e}")
            return rules("Could not reach Gemini, so this is a built-in answer. Check the network connection.")
        return {"reply": reply, "role_used": request.role, "source": "gemini", "notice": None}

    import anthropic
    try:
        reply = _claude_reply(system, messages)
    except anthropic.AuthenticationError:
        return rules("The Anthropic API key was rejected, so this is a built-in answer. Check ANTHROPIC_API_KEY in server/.env.")
    except anthropic.PermissionDeniedError:
        return rules("The Anthropic API key lacks access to this model, so this is a built-in answer.")
    except anthropic.RateLimitError:
        return rules("Claude is rate-limited right now, so this is a built-in answer. Try again shortly.")
    except anthropic.APIStatusError as e:
        print(f"[chat] Claude API error {e.status_code}: {e.message}")
        return rules(f"Claude returned an error ({e.status_code}), so this is a built-in answer.")
    except anthropic.APIConnectionError:
        return rules("Could not reach Claude, so this is a built-in answer. Check the network connection.")
    return {"reply": reply, "role_used": request.role, "source": "claude", "notice": None}
