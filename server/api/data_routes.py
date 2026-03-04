from fastapi import APIRouter, HTTPException
from api.dataset_merger import process_and_merge_datasets

router = APIRouter()
_df = process_and_merge_datasets()

@router.get("/patients")
def get_patients():
    if _df.empty:
        return {"patients": []}
    return {"patients": sorted(_df["Id"].unique().tolist())}

@router.get("/patients/{patient_id}/health_data")
def get_health_data(patient_id: int):
    if _df.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    patient_df = _df[_df["Id"] == patient_id].copy().fillna(0)
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    def col(row, name, default=0.0):
        return float(row[name]) if name in row.index else default

    time_series = []
    for _, row in patient_df.iterrows():
        time_series.append({
            "date":                 str(row["ActivityDate"]).split(" ")[0],
            "TotalSteps":           col(row, "TotalSteps"),
            "Calories":             col(row, "Calories"),
            "TotalDistance":        col(row, "TotalDistance"),
            "TotalMinutesAsleep":   col(row, "TotalMinutesAsleep"),
            "TotalTimeInBed":       col(row, "TotalTimeInBed"),
            "SleepQualityScore":    col(row, "SleepQualityScore", 1.0),
            "AverageHeartrate":     col(row, "AverageHeartrate", 72.0),
            "MaxHeartrate":         col(row, "MaxHeartrate", 100.0),
            "MinHeartrate":         col(row, "MinHeartrate", 60.0),
            "WeightKg":             col(row, "WeightKg", 70.0),
            "BMI":                  col(row, "BMI", 22.0),
            "VeryActiveMinutes":    col(row, "VeryActiveMinutes"),
            "FairlyActiveMinutes":  col(row, "FairlyActiveMinutes"),
            "LightlyActiveMinutes": col(row, "LightlyActiveMinutes"),
            "SedentaryMinutes":     col(row, "SedentaryMinutes"),
            "DailyAvgIntensity":    col(row, "DailyAvgIntensity"),
            "PeakHourlySteps":      col(row, "PeakHourlySteps"),
            "PeakHourlyCalories":   col(row, "PeakHourlyCalories"),
            "AvgMETs":              col(row, "AvgMETs", 10.0),
        })

    def pct(a, b):
        # Use only non-zero start values to avoid -100% from missing last-day data
        if a == 0:
            return 0.0
        return round(((b - a) / a) * 100, 2)

    def first_nonzero(series_key):
        """Get the first non-zero value from time_series for a given key."""
        for entry in time_series:
            if entry.get(series_key, 0) != 0:
                return entry[series_key]
        return 0.0

    def last_nonzero(series_key):
        """Get the last non-zero value from time_series for a given key."""
        for entry in reversed(time_series):
            if entry.get(series_key, 0) != 0:
                return entry[series_key]
        return 0.0

    metrics_change = {
        "steps_change_pct":    0.0,
        "calories_change_pct": 0.0,
        "distance_change_pct": 0.0,
        "sleep_change_pct":    0.0,
    }

    if len(time_series) > 1:
        metrics_change["steps_change_pct"]    = pct(first_nonzero("TotalSteps"),        last_nonzero("TotalSteps"))
        metrics_change["calories_change_pct"] = pct(first_nonzero("Calories"),          last_nonzero("Calories"))
        metrics_change["distance_change_pct"] = pct(first_nonzero("TotalDistance"),     last_nonzero("TotalDistance"))
        metrics_change["sleep_change_pct"]    = pct(first_nonzero("TotalMinutesAsleep"),last_nonzero("TotalMinutesAsleep"))

    return {
        "patient_id":     patient_id,
        "time_series":    time_series,
        "metrics_change": metrics_change,
    }