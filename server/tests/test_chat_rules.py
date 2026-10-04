import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import chat_routes


@pytest.fixture
def client(monkeypatch):
    """Chat API over a tiny in-memory dataset, with no LLM credentials."""
    rows = []
    for i, steps in enumerate([4000, 5000, 6000]):
        rows.append({
            "Id": 1, "ActivityDate": pd.Timestamp("2016-04-12") + pd.Timedelta(days=i),
            "TotalSteps": steps, "Calories": 1900, "VeryActiveMinutes": 5, "FairlyActiveMinutes": 5,
            "SedentaryMinutes": 800, "AvgMETs": 11.0, "PeakHourlySteps": 900,
            "TotalMinutesAsleep": 0, "SleepQualityScore": 1.0, "SleepLogged": False,
            "AverageHeartrate": 72.0, "HeartRateMeasured": False,
            "WeightKg": 70.0, "BMI": 22.0, "WeightMeasured": False,
        })
    monkeypatch.setattr(chat_routes, "process_and_merge_datasets", lambda: pd.DataFrame(rows))
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    app = FastAPI()
    app.include_router(chat_routes.router, prefix="/api")
    return TestClient(app)


def ask(client, message, role="doctor"):
    res = client.post("/api/chat", json={"message": message, "role": role,
                                          "patient_context": {"patient_id": 1}})
    assert res.status_code == 200
    return res.json()


def test_status_reports_rules_mode_without_key(client):
    assert client.get("/api/chat/status").json() == {"llm_enabled": False, "provider": None, "model": None}


def test_falls_back_to_rules_without_key(client):
    body = ask(client, "How many steps?")
    assert body["source"] == "rules"
    assert "5,000 steps/day" in body["reply"]


def test_placeholders_are_not_reported_as_readings(client):
    assert "No heart-rate data" in ask(client, "What is the heart rate?")["reply"]
    assert "No sleep was logged" in ask(client, "How is their sleep?")["reply"]
    assert "No weight" in ask(client, "What about BMI?")["reply"]
    assert "72" not in ask(client, "Give me a summary")["reply"]


def test_patient_role_gets_patient_wording(client):
    assert ask(client, "How many steps?", role="patient")["reply"].startswith("You are averaging")


def test_history_is_accepted(client):
    res = client.post("/api/chat", json={
        "message": "and sleep?", "role": "doctor", "patient_context": {"patient_id": 1},
        "history": [{"role": "user", "content": "steps?"}, {"role": "assistant", "content": "5,000"}],
    })
    assert res.status_code == 200


def test_gemini_key_selects_gemini(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    status = client.get("/api/chat/status").json()
    assert status["llm_enabled"] is True and status["provider"] == "gemini"


def test_gemini_reply_is_used_and_gets_history(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}
    def fake_reply(system, messages):
        seen["system"], seen["messages"] = system, messages
        return "From Gemini"
    monkeypatch.setattr(chat_routes, "_gemini_reply", fake_reply)
    monkeypatch.setattr(chat_routes, "_patient_data", lambda pid, stats: "PATIENT-DATA")
    res = client.post("/api/chat", json={
        "message": "and sleep?", "role": "doctor", "patient_context": {"patient_id": 1},
        "history": [{"role": "user", "content": "steps?"}, {"role": "assistant", "content": "5,000"}],
    }).json()
    assert res["source"] == "gemini" and res["reply"] == "From Gemini"
    assert "PATIENT-DATA" in seen["system"] and "clinician" in seen["system"]
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]


def test_gemini_failure_falls_back_to_rules(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    def boom(system, messages):
        raise RuntimeError("network down")
    monkeypatch.setattr(chat_routes, "_gemini_reply", boom)
    monkeypatch.setattr(chat_routes, "_patient_data", lambda pid, stats: "PATIENT-DATA")
    body = ask(client, "How many steps?")
    assert body["source"] == "rules" and "5,000 steps/day" in body["reply"]
    assert "Gemini" in body["notice"]


def test_busy_gemini_model_is_skipped_for_the_next(monkeypatch):
    from google.genai import errors
    monkeypatch.setattr(chat_routes, "GEMINI_MODELS", ["busy-model", "free-model", "unused-model"])
    tried = []
    def fake_generate(model, system, messages):
        tried.append(model)
        if model == "busy-model":
            raise errors.APIError(503, {"error": {"message": "high demand"}})
        return f"answer from {model}"
    monkeypatch.setattr(chat_routes, "_gemini_generate", fake_generate)
    assert chat_routes._gemini_reply("sys", []) == "answer from free-model"
    assert tried == ["busy-model", "free-model"]


def test_rejected_key_does_not_try_other_models(monkeypatch):
    from google.genai import errors
    monkeypatch.setattr(chat_routes, "GEMINI_MODELS", ["a", "b"])
    tried = []
    def fake_generate(model, system, messages):
        tried.append(model)
        raise errors.APIError(400, {"error": {"message": "API key not valid"}})
    monkeypatch.setattr(chat_routes, "_gemini_generate", fake_generate)
    with pytest.raises(errors.APIError):
        chat_routes._gemini_reply("sys", [])
    assert tried == ["a"]
