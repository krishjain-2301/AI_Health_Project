import os
import pandas as pd
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Fitabase Data 4.12.16-5.12.16")

@lru_cache(maxsize=1)
def process_and_merge_datasets() -> pd.DataFrame:
    print(f"[dataset_merger] Loading from: {DATA_DIR}")

    # 1. Base: Daily Activity (required)
    # NOTE: dailyActivity already contains VeryActiveMinutes, FairlyActiveMinutes,
    # LightlyActiveMinutes, SedentaryMinutes — no need to merge dailyIntensities separately
    activity_path = os.path.join(DATA_DIR, "dailyActivity_merged.csv")
    if not os.path.exists(activity_path):
        raise FileNotFoundError(f"dailyActivity_merged.csv not found at: {activity_path}")
    df_activity = pd.read_csv(activity_path)
    df_activity["ActivityDate"] = pd.to_datetime(df_activity["ActivityDate"], format="mixed")

    # 2. Sleep (sleepDay)
    try:
        df_sleep = pd.read_csv(os.path.join(DATA_DIR, "sleepDay_merged.csv"))
        df_sleep["ActivityDate"] = pd.to_datetime(df_sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_sleep_agg = df_sleep.groupby(["Id","ActivityDate"])[["TotalMinutesAsleep","TotalTimeInBed"]].mean().reset_index()
    except Exception as e:
        print(f"[dataset_merger] Sleep unavailable: {e}")
        df_sleep_agg = pd.DataFrame(columns=["Id","ActivityDate","TotalMinutesAsleep","TotalTimeInBed"])

    # 3. Minute Sleep (sleep quality score per day)
    # value: 1=asleep, 2=restless, 3=awake — lower avg = better quality
    try:
        df_min_sleep = pd.read_csv(os.path.join(DATA_DIR, "minuteSleep_merged.csv"))
        df_min_sleep["ActivityDate"] = pd.to_datetime(df_min_sleep["date"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_sleep_quality = df_min_sleep.groupby(["Id","ActivityDate"])["value"].mean().reset_index()
        df_sleep_quality.rename(columns={"value": "SleepQualityScore"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Minute sleep unavailable: {e}")
        df_sleep_quality = pd.DataFrame(columns=["Id","ActivityDate","SleepQualityScore"])

    # 4. Heart Rate
    try:
        df_hr = pd.read_csv(os.path.join(DATA_DIR, "heartrate_seconds_merged.csv"))
        df_hr["ActivityDate"] = pd.to_datetime(df_hr["Time"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_hr_agg = df_hr.groupby(["Id","ActivityDate"])["Value"].agg(
            AverageHeartrate="mean", MaxHeartrate="max", MinHeartrate="min"
        ).reset_index()
        print("[dataset_merger] Heart rate loaded.")
    except Exception as e:
        print(f"[dataset_merger] Heart rate unavailable (default 72bpm): {e}")
        df_hr_agg = pd.DataFrame(columns=["Id","ActivityDate","AverageHeartrate","MaxHeartrate","MinHeartrate"])

    # 5. Weight
    try:
        df_weight = pd.read_csv(os.path.join(DATA_DIR, "weightLogInfo_merged.csv"))
        df_weight["ActivityDate"] = pd.to_datetime(df_weight["Date"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_weight_agg = df_weight.groupby(["Id","ActivityDate"])[["WeightKg","BMI"]].mean().reset_index()
    except Exception as e:
        print(f"[dataset_merger] Weight unavailable: {e}")
        df_weight_agg = pd.DataFrame(columns=["Id","ActivityDate","WeightKg","BMI"])

    # 6. Hourly Steps (peak hour detection)
    try:
        df_hsteps = pd.read_csv(os.path.join(DATA_DIR, "hourlySteps_merged.csv"))
        df_hsteps["ActivityDate"] = pd.to_datetime(df_hsteps["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_hsteps_agg = df_hsteps.groupby(["Id","ActivityDate"])["StepTotal"].max().reset_index()
        df_hsteps_agg.rename(columns={"StepTotal": "PeakHourlySteps"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Hourly steps unavailable: {e}")
        df_hsteps_agg = pd.DataFrame(columns=["Id","ActivityDate","PeakHourlySteps"])

    # 7. Hourly Calories
    try:
        df_hcal = pd.read_csv(os.path.join(DATA_DIR, "hourlyCalories_merged.csv"))
        df_hcal["ActivityDate"] = pd.to_datetime(df_hcal["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_hcal_agg = df_hcal.groupby(["Id","ActivityDate"])["Calories"].max().reset_index()
        df_hcal_agg.rename(columns={"Calories": "PeakHourlyCalories"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Hourly calories unavailable: {e}")
        df_hcal_agg = pd.DataFrame(columns=["Id","ActivityDate","PeakHourlyCalories"])

    # 8. Hourly Intensities
    try:
        df_hint = pd.read_csv(os.path.join(DATA_DIR, "hourlyIntensities_merged.csv"))
        df_hint["ActivityDate"] = pd.to_datetime(df_hint["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
        df_hint_agg = df_hint.groupby(["Id","ActivityDate"])["AverageIntensity"].mean().reset_index()
        df_hint_agg.rename(columns={"AverageIntensity": "DailyAvgIntensity"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Hourly intensities unavailable: {e}")
        df_hint_agg = pd.DataFrame(columns=["Id","ActivityDate","DailyAvgIntensity"])

    # 9. Minute-level Narrow CSVs (real filenames)
    minute_files = {
        "minuteCaloriesNarrow_merged.csv":    ("ActivityMinute", "Calories",  "AvgMinuteCalories"),
        "minuteIntensitiesNarrow_merged.csv": ("ActivityMinute", "Intensity", "AvgMinuteIntensity"),
        "minuteMETsNarrow_merged.csv":        ("ActivityMinute", "METs",      "AvgMETs"),
        "minuteStepsNarrow_merged.csv":       ("ActivityMinute", "Steps",     "AvgMinuteSteps"),
    }
    minute_dfs = []
    for filename, (time_col, value_col, agg_name) in minute_files.items():
        try:
            df_min = pd.read_csv(os.path.join(DATA_DIR, filename))
            df_min["ActivityDate"] = pd.to_datetime(df_min[time_col], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
            df_agg = df_min.groupby(["Id","ActivityDate"])[value_col].mean().reset_index()
            df_agg.rename(columns={value_col: agg_name}, inplace=True)
            minute_dfs.append(df_agg)
            print(f"[dataset_merger] Loaded {filename} -> {agg_name}")
        except Exception as e:
            print(f"[dataset_merger] {filename} unavailable: {e}")

    # Merge all via left joins on base activity
    merged = df_activity.copy()
    for df_to_merge in [
        df_sleep_agg, df_sleep_quality,
        df_hr_agg, df_weight_agg, df_hsteps_agg,
        df_hcal_agg, df_hint_agg
    ] + minute_dfs:
        if not df_to_merge.empty and "ActivityDate" in df_to_merge.columns:
            merged = pd.merge(merged, df_to_merge, on=["Id","ActivityDate"], how="left")

    # Fill missing values with physiological defaults
    merged["TotalMinutesAsleep"] = merged["TotalMinutesAsleep"].fillna(0)
    if "TotalTimeInBed"     in merged.columns: merged["TotalTimeInBed"]     = merged["TotalTimeInBed"].fillna(0)
    if "SleepQualityScore"  in merged.columns: merged["SleepQualityScore"]  = merged["SleepQualityScore"].fillna(1.0)
    if "DailyAvgIntensity"  in merged.columns: merged["DailyAvgIntensity"]  = merged["DailyAvgIntensity"].fillna(0)
    if "PeakHourlySteps"    in merged.columns: merged["PeakHourlySteps"]    = merged["PeakHourlySteps"].fillna(0)
    if "PeakHourlyCalories" in merged.columns: merged["PeakHourlyCalories"] = merged["PeakHourlyCalories"].fillna(0)
    if "AvgMETs"            in merged.columns: merged["AvgMETs"]            = merged["AvgMETs"].fillna(10.0)

    merged["AverageHeartrate"] = merged.groupby("Id")["AverageHeartrate"].transform(lambda g: g.ffill().bfill()).fillna(72.0)
    if "MaxHeartrate" in merged.columns:
        merged["MaxHeartrate"] = merged.groupby("Id")["MaxHeartrate"].transform(lambda g: g.ffill().bfill()).fillna(100.0)
    if "MinHeartrate" in merged.columns:
        merged["MinHeartrate"] = merged.groupby("Id")["MinHeartrate"].transform(lambda g: g.ffill().bfill()).fillna(60.0)
    merged["WeightKg"] = merged.groupby("Id")["WeightKg"].transform(lambda g: g.ffill().bfill()).fillna(70.0)
    if "BMI" in merged.columns:
        merged["BMI"] = merged.groupby("Id")["BMI"].transform(lambda g: g.ffill().bfill()).fillna(22.0)

    merged = merged.sort_values(["Id","ActivityDate"]).reset_index(drop=True)
    n_patients = merged["Id"].nunique()
    print(f"[dataset_merger] Ready: {len(merged)} rows, {n_patients} patients, {len(merged.columns)} features.")
    return merged