import os
import hashlib
import threading
import numpy as np
import pandas as pd
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Fitabase Data 4.12.16-5.12.16")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Physiological defaults used when a metric was never recorded. Rows filled with
# these are marked via the *Measured / *Logged flag columns so callers can tell
# placeholders from real readings.
DEFAULT_HEARTRATE = 72.0
DEFAULT_MAX_HEARTRATE = 100.0
DEFAULT_MIN_HEARTRATE = 60.0
DEFAULT_WEIGHT_KG = 70.0
DEFAULT_BMI = 22.0
DEFAULT_METS = 10.0
DEFAULT_SLEEP_QUALITY = 1.0

OPTIONAL_COLUMNS = [
    "TotalMinutesAsleep", "TotalTimeInBed", "SleepQualityScore",
    "AverageHeartrate", "MaxHeartrate", "MinHeartrate",
    "WeightKg", "BMI", "PeakHourlySteps", "PeakHourlyCalories",
    "DailyAvgIntensity", "AvgMETs",
]


def merge_from_csvs(data_dir: str) -> pd.DataFrame:
    """Build one row per (Id, ActivityDate) from the Fitabase CSV export in data_dir."""
    print(f"[dataset_merger] Loading from: {data_dir}")

    # 1. Base: Daily Activity (required)
    # NOTE: dailyActivity already contains VeryActiveMinutes, FairlyActiveMinutes,
    # LightlyActiveMinutes, SedentaryMinutes — no need to merge dailyIntensities separately
    activity_path = os.path.join(data_dir, "dailyActivity_merged.csv")
    if not os.path.exists(activity_path):
        raise FileNotFoundError(f"dailyActivity_merged.csv not found at: {activity_path}")
    df_activity = pd.read_csv(activity_path)
    df_activity["ActivityDate"] = pd.to_datetime(df_activity["ActivityDate"], format="mixed")

    # 2. Sleep (sleepDay)
    try:
        df_sleep = pd.read_csv(os.path.join(data_dir, "sleepDay_merged.csv"))
        df_sleep["ActivityDate"] = pd.to_datetime(df_sleep["SleepDay"], format=TS_FORMAT).dt.normalize()
        df_sleep_agg = df_sleep.groupby(["Id","ActivityDate"])[["TotalMinutesAsleep","TotalTimeInBed"]].mean().reset_index()
    except Exception as e:
        print(f"[dataset_merger] Sleep unavailable: {e}")
        df_sleep_agg = pd.DataFrame(columns=["Id","ActivityDate","TotalMinutesAsleep","TotalTimeInBed"])

    # 3. Minute Sleep (sleep quality score per day)
    # value: 1=asleep, 2=restless, 3=awake — lower avg = better quality
    try:
        df_min_sleep = pd.read_csv(os.path.join(data_dir, "minuteSleep_merged.csv"))
        df_min_sleep["ActivityDate"] = pd.to_datetime(df_min_sleep["date"], format=TS_FORMAT).dt.normalize()
        df_sleep_quality = df_min_sleep.groupby(["Id","ActivityDate"])["value"].mean().reset_index()
        df_sleep_quality.rename(columns={"value": "SleepQualityScore"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Minute sleep unavailable: {e}")
        df_sleep_quality = pd.DataFrame(columns=["Id","ActivityDate","SleepQualityScore"])

    # 4. Heart Rate
    try:
        df_hr = pd.read_csv(os.path.join(data_dir, "heartrate_seconds_merged.csv"))
        df_hr["ActivityDate"] = pd.to_datetime(df_hr["Time"], format=TS_FORMAT).dt.normalize()
        df_hr_agg = df_hr.groupby(["Id","ActivityDate"])["Value"].agg(
            AverageHeartrate="mean", MaxHeartrate="max", MinHeartrate="min"
        ).reset_index()
        print("[dataset_merger] Heart rate loaded.")
    except Exception as e:
        print(f"[dataset_merger] Heart rate unavailable (default 72bpm): {e}")
        df_hr_agg = pd.DataFrame(columns=["Id","ActivityDate","AverageHeartrate","MaxHeartrate","MinHeartrate"])

    # 5. Weight
    try:
        df_weight = pd.read_csv(os.path.join(data_dir, "weightLogInfo_merged.csv"))
        df_weight["ActivityDate"] = pd.to_datetime(df_weight["Date"], format=TS_FORMAT).dt.normalize()
        df_weight_agg = df_weight.groupby(["Id","ActivityDate"])[["WeightKg","BMI"]].mean().reset_index()
    except Exception as e:
        print(f"[dataset_merger] Weight unavailable: {e}")
        df_weight_agg = pd.DataFrame(columns=["Id","ActivityDate","WeightKg","BMI"])

    # 6. Hourly Steps (peak hour detection)
    try:
        df_hsteps = pd.read_csv(os.path.join(data_dir, "hourlySteps_merged.csv"))
        df_hsteps["ActivityDate"] = pd.to_datetime(df_hsteps["ActivityHour"], format=TS_FORMAT).dt.normalize()
        df_hsteps_agg = df_hsteps.groupby(["Id","ActivityDate"])["StepTotal"].max().reset_index()
        df_hsteps_agg.rename(columns={"StepTotal": "PeakHourlySteps"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Hourly steps unavailable: {e}")
        df_hsteps_agg = pd.DataFrame(columns=["Id","ActivityDate","PeakHourlySteps"])

    # 7. Hourly Calories
    try:
        df_hcal = pd.read_csv(os.path.join(data_dir, "hourlyCalories_merged.csv"))
        df_hcal["ActivityDate"] = pd.to_datetime(df_hcal["ActivityHour"], format=TS_FORMAT).dt.normalize()
        df_hcal_agg = df_hcal.groupby(["Id","ActivityDate"])["Calories"].max().reset_index()
        df_hcal_agg.rename(columns={"Calories": "PeakHourlyCalories"}, inplace=True)
    except Exception as e:
        print(f"[dataset_merger] Hourly calories unavailable: {e}")
        df_hcal_agg = pd.DataFrame(columns=["Id","ActivityDate","PeakHourlyCalories"])

    # 8. Hourly Intensities
    try:
        df_hint = pd.read_csv(os.path.join(data_dir, "hourlyIntensities_merged.csv"))
        df_hint["ActivityDate"] = pd.to_datetime(df_hint["ActivityHour"], format=TS_FORMAT).dt.normalize()
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
            df_min = pd.read_csv(os.path.join(data_dir, filename))
            df_min["ActivityDate"] = pd.to_datetime(df_min[time_col], format=TS_FORMAT).dt.normalize()
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

    for col in OPTIONAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged.sort_values(["Id","ActivityDate"]).reset_index(drop=True)

    # Record what was really measured before any placeholder is filled in
    merged["HeartRateMeasured"] = merged["AverageHeartrate"].notna()
    merged["WeightMeasured"]    = merged["WeightKg"].notna()
    merged["SleepLogged"]       = merged["TotalMinutesAsleep"].notna()

    # Fill missing values with physiological defaults
    merged["TotalMinutesAsleep"] = merged["TotalMinutesAsleep"].fillna(0)
    merged["TotalTimeInBed"]     = merged["TotalTimeInBed"].fillna(0)
    merged["SleepQualityScore"]  = merged["SleepQualityScore"].fillna(DEFAULT_SLEEP_QUALITY)
    merged["DailyAvgIntensity"]  = merged["DailyAvgIntensity"].fillna(0)
    merged["PeakHourlySteps"]    = merged["PeakHourlySteps"].fillna(0)
    merged["PeakHourlyCalories"] = merged["PeakHourlyCalories"].fillna(0)
    merged["AvgMETs"]            = merged["AvgMETs"].fillna(DEFAULT_METS)

    def carry(col, default):
        return merged.groupby("Id")[col].transform(lambda g: g.ffill().bfill()).fillna(default)

    merged["AverageHeartrate"] = carry("AverageHeartrate", DEFAULT_HEARTRATE)
    merged["MaxHeartrate"]     = carry("MaxHeartrate", DEFAULT_MAX_HEARTRATE)
    merged["MinHeartrate"]     = carry("MinHeartrate", DEFAULT_MIN_HEARTRATE)
    merged["WeightKg"]         = carry("WeightKg", DEFAULT_WEIGHT_KG)
    merged["BMI"]              = carry("BMI", DEFAULT_BMI)

    n_patients = merged["Id"].nunique()
    print(f"[dataset_merger] Ready: {len(merged)} rows, {n_patients} patients, {len(merged.columns)} features.")
    return merged


def _cache_path(data_dir: str) -> str:
    """Cache file name that changes whenever any CSV in data_dir changes."""
    h = hashlib.sha1()
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".csv"):
            st = os.stat(os.path.join(data_dir, name))
            h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
    # v2: bump when merge_from_csvs changes its output columns
    return os.path.join(CACHE_DIR, f"merged_v2_{h.hexdigest()[:16]}.pkl")


_load_lock = threading.Lock()


def process_and_merge_datasets() -> pd.DataFrame:
    # Requests can arrive while the startup warm-up is still loading; the lock
    # makes them wait for that one load instead of each starting their own.
    with _load_lock:
        return _load_merged()


@lru_cache(maxsize=1)
def _load_merged() -> pd.DataFrame:
    # Parsing the second-level heart-rate file takes ~30s, so keep the merged
    # frame on disk between server restarts.
    cache_path = None
    try:
        cache_path = _cache_path(DATA_DIR)
        if os.path.exists(cache_path):
            merged = pd.read_pickle(cache_path)
            print(f"[dataset_merger] Loaded cached merge: {len(merged)} rows.")
            return merged
    except Exception as e:
        print(f"[dataset_merger] Cache unusable, rebuilding: {e}")

    merged = merge_from_csvs(DATA_DIR)

    if cache_path:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            merged.to_pickle(cache_path)
        except Exception as e:
            print(f"[dataset_merger] Could not write cache: {e}")
    return merged
