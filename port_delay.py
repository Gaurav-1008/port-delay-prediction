"""Reproducible synthetic port queues, interpretable classification, and prediction CLI."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DELAY_HOURS = 6.0
RAW_INPUTS = [
    "queue_length", "operational_berths", "busy_berths", "arrivals_past_6h",
    "weather_severity", "past_avg_service_hours",
]
FEATURES = [
    "queue_per_berth", "berth_utilization", "arrival_pressure",
    "weather_severity", "past_avg_service_hours",
]
PORTS = {"PORT_A": (3, 7.0), "PORT_B": (5, 8.0), "PORT_C": (7, 8.5)}
FEATURE_LABELS = {
    "queue_per_berth": "Waiting vessels per operational berth",
    "berth_utilization": "Fraction of operational berths occupied",
    "arrival_pressure": "Arrivals in the last six hours per berth",
    "weather_severity": "Weather severity (0 clear, 1 moderate, 2 severe)",
    "past_avg_service_hours": "Past average completed service time (hours)",
}


def simulate(n_records=500, seed=SEED, warmup=150, missing_rate=0.03):
    """First-come-first-served queues; sample hidden service times, export arrival snapshots."""
    if n_records < 1 or warmup < 0 or not 0 <= missing_rate < 1:
        raise ValueError("Require positive record count, nonnegative warmup, and 0 <= missing_rate < 1.")
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    free = {port: [0.0] * capacity for port, (capacity, _) in PORTS.items()}
    visits = {port: [] for port in PORTS}
    weather = {}
    records = []
    hour = 0.0
    for i in range(n_records + warmup):
        clock = hour % 24
        rate = 1.35 * (1.45 if 8 <= clock < 18 else 0.72)
        hour += float(rng.exponential(1 / rate))
        port = str(rng.choice(list(PORTS), p=np.array([3, 5, 7]) / 15))
        capacity, base_service = PORTS[port]
        state = visits[port]
        weather_key = (port, int(hour // 12))
        if weather_key not in weather:
            weather[weather_key] = int(rng.choice([0, 1, 2], p=[0.70, 0.23, 0.07]))
        severity = weather[weather_key]
        completed = [visit["service"] for visit in state if visit["end"] <= hour]
        queue = sum(visit["start"] > hour for visit in state)
        busy = sum(visit["start"] <= hour < visit["end"] for visit in state)
        recent = sum(hour - 6 <= visit["arrival"] < hour for visit in state)
        history = float(np.mean(completed)) if completed else base_service

        earliest_free = heapq.heappop(free[port])
        start = max(hour, earliest_free)
        sigma = 0.45
        service = float(rng.lognormal(np.log(base_service) - sigma**2 / 2, sigma))
        service *= 1 + 0.25 * severity
        if rng.random() < 0.04:
            service += float(rng.uniform(4, 16))
        end = start + service
        heapq.heappush(free[port], end)
        state.append({"arrival": hour, "start": start, "end": end, "service": service})

        if i >= warmup:
            arrival_time = origin + pd.to_timedelta(hour, unit="h")
            berth_start = origin + pd.to_timedelta(start, unit="h")
            waiting = (berth_start - arrival_time).total_seconds() / 3600
            delayed = int(waiting > DELAY_HOURS)
            known_at = (
                arrival_time + pd.Timedelta(hours=DELAY_HOURS, microseconds=1)
                if delayed else berth_start
            )
            records.append({
                "visit_id": f"VISIT_{i - warmup + 1:04d}",
                "port_id": port,
                "arrival_time": arrival_time,
                "berth_start_time": berth_start,
                "service_end_time": origin + pd.to_timedelta(end, unit="h"),
                "label_available_time": known_at,
                "queue_length": queue,
                "operational_berths": capacity,
                "busy_berths": busy,
                "arrivals_past_6h": recent,
                "weather_severity": severity,
                "past_avg_service_hours": history,
                "waiting_hours": waiting,
                "is_delayed": delayed,
            })
    data = pd.DataFrame(records)
    for name in ["weather_severity", "past_avg_service_hours"]:
        data.loc[rng.random(n_records) < missing_rate, name] = np.nan
    return data


def feature_frame(observations):
    """Validate observed inputs and derive features without reading outcome columns."""
    missing = set(RAW_INPUTS) - set(observations.columns)
    if missing:
        raise ValueError(f"Missing required input fields: {', '.join(sorted(missing))}")
    raw = observations[RAW_INPUTS].copy()
    for name in RAW_INPUTS:
        if raw[name].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{name} must be numeric, not boolean.")
        raw[name] = pd.to_numeric(raw[name], errors="raise")
        if np.isinf(raw[name].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} must be finite.")
    counts = RAW_INPUTS[:4]
    if raw[counts].isna().any().any() or (raw[counts] < 0).any().any():
        raise ValueError("Queue, berth, and arrival counts must be present and nonnegative.")
    if (raw[counts] % 1 != 0).any().any():
        raise ValueError("Queue, berth, and arrival counts must be integers.")
    if (raw.operational_berths < 1).any() or (raw.busy_berths > raw.operational_berths).any():
        raise ValueError("At least one operational berth is required; busy berths cannot exceed capacity.")
    if not raw.weather_severity.dropna().isin([0, 1, 2]).all():
        raise ValueError("Weather severity must be 0, 1, 2, or null.")
    if (raw.past_avg_service_hours.dropna() <= 0).any():
        raise ValueError("Historical service hours must be positive or null.")
    return pd.DataFrame({
        "queue_per_berth": raw.queue_length / raw.operational_berths,
        "berth_utilization": raw.busy_berths / raw.operational_berths,
        "arrival_pressure": raw.arrivals_past_6h / raw.operational_berths,
        "weather_severity": raw.weather_severity,
        "past_avg_service_hours": raw.past_avg_service_hours,
    }, index=observations.index)[FEATURES]


def temporal_split(data):
    if not data.arrival_time.is_monotonic_increasing or data.arrival_time.duplicated().any():
        raise ValueError("Visits must have unique arrival times in ascending order.")
    first, second = int(len(data) * 0.7), int(len(data) * 0.8)
    validation_start, test_start = data.iloc[first].arrival_time, data.iloc[second].arrival_time
    train_candidates, validation_candidates = data.iloc[:first], data.iloc[first:second]
    train = train_candidates[train_candidates.label_available_time < validation_start]
    validation = validation_candidates[validation_candidates.label_available_time < test_start]
    test = data.iloc[second:]
    for name, partition in [("train", train), ("validation", validation), ("test", test)]:
        if partition.is_delayed.nunique() != 2:
            raise ValueError(f"{name} must contain both classes for this evaluation.")
    details = {
        "train": len(train), "validation": len(validation), "test": len(test),
        "train_unresolved_excluded": len(train_candidates) - len(train),
        "validation_unresolved_excluded": len(validation_candidates) - len(validation),
        "validation_start": validation_start.isoformat(), "test_start": test_start.isoformat(),
    }
    return train, validation, test, details


def score(y, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def predict_one(artifact, observation):
    features = feature_frame(pd.DataFrame([observation]))
    pipeline = artifact["pipeline"]
    probability = float(pipeline.predict_proba(features)[0, 1])
    delayed = probability >= artifact["threshold"]
    imputer, scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    imputed = imputer.transform(features)
    standardized = scaler.transform(imputed)[0]
    contributions = standardized * classifier.coef_[0]
    factors = []
    for index, name in enumerate(FEATURES):
        value = features.iloc[0][name]
        factors.append({
            "feature": name,
            "description": FEATURE_LABELS[name],
            "value": None if pd.isna(value) else float(value),
            "imputed": bool(pd.isna(value)),
            "contribution_log_odds": float(contributions[index]),
            "effect": "increases delay risk" if contributions[index] >= 0 else "decreases delay risk",
        })
    factors.sort(key=lambda factor: abs(factor["contribution_log_odds"]), reverse=True)
    return {
        "prediction": "delay_over_6_hours" if delayed else "no_delay_over_6_hours",
        "delay_probability": probability,
        "confidence_score": probability if delayed else 1 - probability,
        "confidence_definition": "Estimated probability of the selected outcome, not measured accuracy.",
        "decision_threshold": artifact["threshold"],
        "main_factors": factors[:3],
        "factor_contributions": factors,
        "reference_log_odds": float(classifier.intercept_[0]),
        "missing_fields_imputed": [
            name for name in RAW_INPUTS if observation.get(name) is None or pd.isna(observation[name])
        ],
        "suggested_action": (
            "Review berth scheduling and notify downstream planners of potential disruption."
            if delayed else "Maintain the current plan and continue monitoring arrival conditions."
        ),
        "scope": "Synthetic-data prototype; real-port probability reliability is unverified.",
    }


def save_figures(data, test, probabilities, pipeline, results):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    colors = ["#24758c", "#dd8d3e"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axes[0].hist(data.waiting_hours, bins=24, color=colors[0], edgecolor="white")
    axes[0].axvline(DELAY_HOURS, color=colors[1], linestyle="--", label="6-hour threshold")
    axes[0].set(xlabel="Wait before berthing (hours)", ylabel="Vessel visits", title="Simulated waiting times")
    axes[0].legend(fontsize=8)
    data.groupby("queue_length").is_delayed.agg(["mean", "size"]).plot(
        y="mean", ax=axes[1], legend=False, color=colors[0], marker="o",
    )
    axes[1].set(xlabel="Vessels waiting at arrival", ylabel="Fraction delayed", title="Queue pressure and delay", ylim=(-0.05, 1.05))
    data.groupby("weather_severity").is_delayed.mean().plot.bar(ax=axes[2], color=colors[0], rot=0)
    axes[2].set(xlabel="Weather severity (observed)", ylabel="Fraction delayed", title="Weather and delay", ylim=(0, 1))
    fig.savefig(results / "data_overview.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    predicted = (probabilities >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        test.is_delayed, predicted, display_labels=["No delay", "Delayed"],
        cmap="Blues", colorbar=False, ax=axes[0],
    )
    axes[0].set_title(f"Held-out confusion matrix (n={len(test)})")
    for name, p, color in [
        ("Logistic regression", probabilities, colors[0]),
        ("Training-prior baseline", np.repeat(float(pipeline["baseline_probability"]), len(test)), colors[1]),
    ]:
        observed, mean_probability = calibration_curve(test.is_delayed, p, n_bins=4, strategy="quantile")
        axes[1].plot(mean_probability, observed, "o-", label=name, color=color)
    axes[1].plot([0, 1], [0, 1], "--", color="#8795a1", linewidth=1)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed delay fraction",
                title="Reliability: 4 quantile bins", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(results / "evaluation.png", dpi=180)
    plt.close(fig)

    classifier = pipeline["model"].steps[-1][1]
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    coefficients = pd.Series(classifier.coef_[0], index=FEATURES).sort_values()
    ax.barh([FEATURE_LABELS[key] for key in coefficients.index], coefficients,
            color=[colors[0] if value >= 0 else colors[1] for value in coefficients])
    ax.axvline(0, color="#8795a1", linewidth=1)
    ax.set(xlabel="Log-odds coefficient (per training SD)", title="Global model associations")
    fig.savefig(results / "feature_coefficients.png", dpi=170)
    plt.close(fig)


def run(project):
    project = Path(project)
    for folder in ["data", "results", "examples"]:
        (project / folder).mkdir(parents=True, exist_ok=True)
    results = project / "results"
    data = simulate()
    feature_frame(data)
    if data.visit_id.duplicated().any() or data.waiting_hours.lt(0).any():
        raise ValueError("Invalid simulated visits.")
    data_path = project / "data/synthetic_port_calls.csv"
    data.to_csv(data_path, index=False)
    train, validation, test, split = temporal_split(data)
    x_train, x_validation, x_test = [feature_frame(frame) for frame in [train, validation, test]]
    candidates = []
    models = {}
    for strength in [0.1, 1.0, 10.0]:
        pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=strength, max_iter=2000, random_state=SEED))
        pipeline.fit(x_train, train.is_delayed)
        metrics = score(validation.is_delayed, pipeline.predict_proba(x_validation)[:, 1])
        candidates.append({"C": strength, **metrics})
        models[strength] = pipeline
    selection = pd.DataFrame(candidates).sort_values(["brier", "C"])
    best_c = float(selection.iloc[0].C)
    pipeline = models[best_c]
    baseline = DummyClassifier(strategy="prior").fit(x_train, train.is_delayed)
    baseline_probability = float(baseline.class_prior_[1])
    test_probabilities = pipeline.predict_proba(x_test)[:, 1]
    rows = []
    for name, estimator in [("training_prior_baseline", baseline), ("logistic_regression", pipeline)]:
        for partition, x, frame in [("validation", x_validation, validation), ("test", x_test, test)]:
            rows.append({"model": name, "split": partition, "n": len(frame),
                         **score(frame.is_delayed, estimator.predict_proba(x)[:, 1])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results / "metrics.csv", index=False)
    selection.to_csv(results / "validation_candidates.csv", index=False)
    artifact = {"pipeline": pipeline, "features": FEATURES, "raw_inputs": RAW_INPUTS,
                "threshold": 0.5, "delay_hours": DELAY_HOURS, "seed": SEED, "selected_C": best_c}
    joblib.dump(artifact, results / "model.joblib")
    predictions = test[["visit_id", "port_id", "arrival_time", "waiting_hours", "is_delayed", *RAW_INPUTS]].copy()
    predictions["delay_probability"] = test_probabilities
    predictions["predicted_delayed"] = (test_probabilities >= 0.5).astype(int)
    predictions.to_csv(results / "test_predictions.csv", index=False)
    reliability = pd.DataFrame({"probability": test_probabilities, "actual": test.is_delayed.to_numpy()})
    reliability["bin"] = pd.qcut(reliability.probability, q=4, duplicates="drop")
    reliability.groupby("bin", observed=True).agg(
        count=("actual", "size"), mean_probability=("probability", "mean"),
        observed_delay_fraction=("actual", "mean"),
    ).to_csv(results / "calibration_bins.csv")
    data.groupby("port_id").agg(visits=("visit_id", "size"), delay_rate=("is_delayed", "mean"),
                                mean_wait_hours=("waiting_hours", "mean")).to_csv(results / "port_summary.csv")
    summary = {
        "seed": SEED, "n_records": len(data), "warmup_visits": 150,
        "delay_threshold_hours": DELAY_HOURS, "decision_threshold": 0.5,
        "selected_C": best_c, "split": split,
        "delayed_visits": int(data.is_delayed.sum()), "delay_rate": float(data.is_delayed.mean()),
        "test_delayed": int(test.is_delayed.sum()),
        "test_confusion_matrix": confusion_matrix(test.is_delayed, test_probabilities >= 0.5).tolist(),
        "missing_values": {key: int(value) for key, value in data[RAW_INPUTS].isna().sum().items()},
        "date_range": [data.arrival_time.min().isoformat(), data.arrival_time.max().isoformat()],
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "port_parameters": {name: {"berths": count, "base_mean_service_hours": service}
                            for name, (count, service) in PORTS.items()},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    examples = {}
    for name, position in [("low_risk", int(np.argmin(test_probabilities))),
                           ("high_risk", int(np.argmax(test_probabilities)))]:
        row = test.iloc[position]
        observation = {key: None if pd.isna(row[key]) else float(row[key]) for key in RAW_INPUTS}
        (project / f"examples/{name}.json").write_text(json.dumps(observation, indent=2, allow_nan=False) + "\n")
        examples[name] = {"source_visit": row.visit_id, "observation": observation,
                          "result": predict_one(artifact, observation)}
    (results / "example_predictions.json").write_text(json.dumps(examples, indent=2, allow_nan=False) + "\n")
    save_figures(data, test, test_probabilities, {"model": pipeline, "baseline_probability": baseline_probability}, results)
    print(metrics.round(4).to_string(index=False))
    print(json.dumps(summary["split"], indent=2))
    return {"data": data, "metrics": metrics, "summary": summary, "model": artifact, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Recreate synthetic data, fit model, and write results.")
    train.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    predict = subparsers.add_parser("predict", help="Predict and explain one arrival from a JSON file.")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "results/model.joblib")
    arguments = parser.parse_args()
    if arguments.command == "train":
        run(arguments.project)
    else:
        try:
            observation = json.loads(arguments.input.read_text())
            if not isinstance(observation, dict):
                raise ValueError("Prediction input must be a JSON object.")
            result = predict_one(joblib.load(arguments.model), observation)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
