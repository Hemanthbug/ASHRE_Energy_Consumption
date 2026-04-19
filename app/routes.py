from flask import Blueprint, jsonify, render_template, request
import pandas as pd
import numpy as np
import joblib

bp = Blueprint("routes", __name__)

PARQUET_PATH = "data/processed/clean_data.parquet"
MODEL_PATH = "models/energy_model.pkl"

FEATURE_COLS = [
    "building_id",
    "meter",
    "square_feet",
    "air_temperature",
    "cloud_coverage",
    "hour",
    "month",
    "dayofweek",
]

model = joblib.load(MODEL_PATH)

@bp.route("/")
def dashboard():
    return render_template("dashboard.html")


@bp.route("/data/<int:building_id>")
def get_data(building_id: int):
    cols = FEATURE_COLS + ["meter_reading"]

    try:
        df = pd.read_parquet(
            PARQUET_PATH,
            columns=cols,
            filters=[("building_id", "=", building_id)]
        )
    except Exception:
        df_all = pd.read_parquet(PARQUET_PATH, columns=cols)
        df = df_all[df_all["building_id"] == building_id]

    if df.empty:
        return jsonify({
            "ok": False,
            "message": f"No data found for building_id={building_id}",
            "kpis": {"total_records": 0, "sample_size": 0, "avg_energy": 0, "max_energy": 0, "waste_rate": 0, "peak_waste_hour": None, "severity": "N/A"},
            "hourly": [],
            "monthly": [],
            "scatter": [],
            "waste_by_dow": [],
            "waste_matrix": [],
            "recommendations": ["Try another building_id (example: 1..1000)."],
            "top_waste_hours": []
        })

    # fast dashboard sample
    sample_n = min(8000, len(df))
    s = df.sample(n=sample_n, random_state=42).copy()

    # predict expected (log) and convert to original
    X = s[FEATURE_COLS]
    pred_log = model.predict(X)
    actual_log = s["meter_reading"].to_numpy()

    expected = np.expm1(pred_log)
    actual = np.expm1(actual_log)
    waste = actual - expected

    s["actual"] = actual
    s["expected"] = expected
    s["waste"] = waste

    # KPIs
    avg_energy = float(np.mean(actual))
    max_energy = float(np.max(actual))

    waste_flag = (waste > (0.25 * np.maximum(expected, 1))) & (waste > 10)
    waste_rate = float(np.mean(waste_flag))

    if waste_rate >= 0.35:
        severity = "High"
    elif waste_rate >= 0.18:
        severity = "Medium"
    else:
        severity = "Low"

    # hourly
    hourly_df = (
        s.groupby("hour")[["actual", "expected", "waste"]]
        .mean()
        .reset_index()
        .sort_values("hour")
    )
    hourly = hourly_df.to_dict(orient="records")

    peak_waste_hour = int(hourly_df.sort_values("waste", ascending=False)["hour"].iloc[0])

    # monthly
    monthly_df = (
        s.groupby("month")[["actual"]]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    monthly = monthly_df.to_dict(orient="records")

    # waste by day of week
    dow_df = (
        s.groupby("dayofweek")[["waste"]]
        .mean()
        .reset_index()
        .sort_values("dayofweek")
    )
    waste_by_dow = dow_df.to_dict(orient="records")

    # Waste hotspot matrix: hour x dayofweek
    matrix_df = (
        s.groupby(["dayofweek", "hour"])[["waste"]]
        .mean()
        .reset_index()
        .sort_values(["dayofweek", "hour"])
    )
    waste_matrix = matrix_df.to_dict(orient="records")

    # temp scatter (temp vs actual)
    scatter_df = s[["air_temperature", "actual"]].dropna()
    scatter_df = scatter_df.sample(n=min(1500, len(scatter_df)), random_state=42)
    scatter = scatter_df.to_dict(orient="records")

    # top waste hours
    top_waste_hours = (
        hourly_df.sort_values("waste", ascending=False)
        .head(5)[["hour", "waste"]]
        .to_dict(orient="records")
    )

    # recommendations (dynamic)
    recs = []
    if severity == "High":
        recs.append("High waste risk → tighten HVAC schedules, check setpoints, verify sensor calibration.")
    elif severity == "Medium":
        recs.append("Moderate waste risk → optimize peak-hour operation, validate weekend/off-hour schedules.")
    else:
        recs.append("Low waste risk → maintain controls, keep monitoring peaks and temperature effects.")

    if peak_waste_hour <= 6 or peak_waste_hour >= 21:
        recs.append("Peak waste occurs off-hours → check overnight lighting/HVAC timers and BAS schedules.")
    else:
        recs.append("Peak waste occurs during working hours → reduce peak loads, tune setpoints, apply staging controls.")

    if scatter_df.shape[0] > 80:
        corr = np.corrcoef(scatter_df["air_temperature"].fillna(0), scatter_df["actual"])[0, 1]
        if np.isfinite(corr) and corr > 0.35:
            recs.append("Energy rises with temperature → focus on cooling efficiency (filters, chiller staging, shading, insulation).")

    return jsonify({
        "ok": True,
        "message": "ok",
        "kpis": {
            "total_records": int(len(df)),
            "sample_size": int(sample_n),
            "avg_energy": avg_energy,
            "max_energy": max_energy,
            "waste_rate": waste_rate,
            "peak_waste_hour": peak_waste_hour,
            "severity": severity
        },
        "hourly": hourly,
        "monthly": monthly,
        "scatter": scatter,
        "waste_by_dow": waste_by_dow,
        "waste_matrix": waste_matrix,
        "recommendations": recs,
        "top_waste_hours": top_waste_hours
    })


@bp.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # DataFrame with names -> removes sklearn warning
    X = pd.DataFrame([{
        "building_id": int(data["building_id"]),
        "meter": int(data["meter"]),
        "square_feet": float(data["square_feet"]),
        "air_temperature": float(data["air_temperature"]),
        "cloud_coverage": float(data["cloud_coverage"]),
        "hour": int(data["hour"]),
        "month": int(data["month"]),
        "dayofweek": int(data["dayofweek"]),
    }], columns=FEATURE_COLS)

    pred_log = model.predict(X)[0]
    predicted = float(np.expm1(pred_log))

    # baseline “typical actual” for same bucket
    try:
        ref = pd.read_parquet(
            PARQUET_PATH,
            columns=["meter_reading"],
            filters=[
                ("building_id", "=", int(data["building_id"])),
                ("meter", "=", int(data["meter"])),
                ("hour", "=", int(data["hour"])),
                ("month", "=", int(data["month"])),
                ("dayofweek", "=", int(data["dayofweek"]))
            ]
        )
        if ref.empty:
            baseline_actual = None
            baseline_n = 0
        else:
            baseline_actual = float(np.expm1(ref["meter_reading"]).median())
            baseline_n = int(len(ref))
    except Exception:
        baseline_actual = None
        baseline_n = 0

    return jsonify({
        "predicted_energy": predicted,
        "baseline_actual": baseline_actual,
        "baseline_n": baseline_n
    })