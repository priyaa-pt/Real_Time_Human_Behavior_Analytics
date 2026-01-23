from fastapi import FastAPI
import pandas as pd
import os

app = FastAPI()

CSV_PATH = os.path.join("logs", "tracking_data.csv")


@app.get("/")
def home():
    return {"message": "Real-Time Human Behavior Analytics API"}


@app.get("/metrics")
def get_metrics():
    if not os.path.exists(CSV_PATH):
        return {"error": "Data not found"}

    df = pd.read_csv(CSV_PATH)

    footfall = df["person_id"].nunique()

    dwell_times = {}
    for person_id, group in df.groupby("person_id"):
        dwell_times[int(person_id)] = round(
            group["timestamp"].max() - group["timestamp"].min(), 2
        )

    avg_dwell = round(sum(dwell_times.values()) / len(dwell_times), 2)

    return {
        "footfall": footfall,
        "average_dwell_time": avg_dwell,
        "per_person_dwell_time": dwell_times
    }
