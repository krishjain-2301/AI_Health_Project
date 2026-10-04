import pandas as pd
import pytest

from api.dataset_merger import merge_from_csvs, DEFAULT_HEARTRATE, DEFAULT_WEIGHT_KG


def _write(tmp_path, name, rows):
    pd.DataFrame(rows).to_csv(tmp_path / name, index=False)


@pytest.fixture
def data_dir(tmp_path):
    """Two patients, three days each. Patient 1 has sleep, heart rate and weight
    on some days; patient 2 has activity only."""
    activity = []
    for pid in (1, 2):
        for day, steps in (("4/12/2016", 9000), ("4/13/2016", 4000), ("4/14/2016", 0)):
            activity.append({
                "Id": pid, "ActivityDate": day, "TotalSteps": steps, "TotalDistance": steps / 1400,
                "VeryActiveMinutes": 20, "FairlyActiveMinutes": 10, "LightlyActiveMinutes": 200,
                "SedentaryMinutes": 700, "Calories": 2000,
            })
    _write(tmp_path, "dailyActivity_merged.csv", activity)
    _write(tmp_path, "sleepDay_merged.csv", [
        {"Id": 1, "SleepDay": "4/12/2016 12:00:00 AM", "TotalSleepRecords": 1,
         "TotalMinutesAsleep": 420, "TotalTimeInBed": 450},
        # duplicate record for the same night must not duplicate the day
        {"Id": 1, "SleepDay": "4/12/2016 12:00:00 AM", "TotalSleepRecords": 1,
         "TotalMinutesAsleep": 400, "TotalTimeInBed": 430},
    ])
    _write(tmp_path, "heartrate_seconds_merged.csv", [
        {"Id": 1, "Time": "4/13/2016 8:00:00 AM", "Value": 60},
        {"Id": 1, "Time": "4/13/2016 8:00:05 AM", "Value": 80},
    ])
    _write(tmp_path, "weightLogInfo_merged.csv", [
        {"Id": 1, "Date": "4/12/2016 11:59:59 PM", "WeightKg": 82.5, "BMI": 26.1},
    ])
    return str(tmp_path)


def test_one_row_per_patient_day(data_dir):
    df = merge_from_csvs(data_dir)
    assert len(df) == 6
    assert not df.duplicated(["Id", "ActivityDate"]).any()


def test_sleep_is_merged_and_flagged(data_dir):
    df = merge_from_csvs(data_dir)
    p1 = df[df["Id"] == 1].reset_index(drop=True)
    assert p1.loc[0, "TotalMinutesAsleep"] == 410          # mean of the two records
    assert p1["SleepLogged"].tolist() == [True, False, False]
    assert p1.loc[1, "TotalMinutesAsleep"] == 0


def test_heart_rate_is_aggregated_then_carried_within_patient(data_dir):
    df = merge_from_csvs(data_dir)
    p1 = df[df["Id"] == 1].reset_index(drop=True)
    assert p1.loc[1, "AverageHeartrate"] == 70
    assert p1.loc[1, "MaxHeartrate"] == 80
    # only the day with readings counts as measured; neighbours are carried values
    assert p1["HeartRateMeasured"].tolist() == [False, True, False]
    assert p1["AverageHeartrate"].tolist() == [70, 70, 70]


def test_placeholders_are_flagged_for_patient_without_readings(data_dir):
    df = merge_from_csvs(data_dir)
    p2 = df[df["Id"] == 2]
    assert (p2["AverageHeartrate"] == DEFAULT_HEARTRATE).all()
    assert (p2["WeightKg"] == DEFAULT_WEIGHT_KG).all()
    assert not p2["HeartRateMeasured"].any()
    assert not p2["WeightMeasured"].any()
    assert not p2["SleepLogged"].any()


def test_weight_is_measured_only_on_logged_day(data_dir):
    df = merge_from_csvs(data_dir)
    p1 = df[df["Id"] == 1].reset_index(drop=True)
    assert p1["WeightMeasured"].tolist() == [True, False, False]
    assert (p1["WeightKg"] == 82.5).all()


def test_optional_files_can_be_missing(tmp_path):
    _write(tmp_path, "dailyActivity_merged.csv", [{
        "Id": 1, "ActivityDate": "4/12/2016", "TotalSteps": 5000, "TotalDistance": 3.5,
        "VeryActiveMinutes": 5, "FairlyActiveMinutes": 5, "LightlyActiveMinutes": 100,
        "SedentaryMinutes": 900, "Calories": 1800,
    }])
    df = merge_from_csvs(str(tmp_path))
    assert len(df) == 1
    assert not df[["AverageHeartrate", "WeightKg", "AvgMETs", "TotalMinutesAsleep"]].isna().any().any()


def test_missing_activity_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_from_csvs(str(tmp_path))
