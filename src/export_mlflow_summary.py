import os
import csv
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
OUTPUT_CSV = "mlflow_summary.csv"
ARTIFACTS_DIR = "mlflow_artifacts"
MLFLOW_TRACKING_URI = None

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
exps = client.list_experiments()

all_param_keys = set()
all_metric_keys = set()
runs_info = []

# Gather runs and keys
for exp in exps:
    runs = client.search_runs(exp.experiment_id)
    for run in runs:
        runs_info.append((exp, run))
        for k in run.data.params.keys():
            all_param_keys.add(k)
        for k in run.data.metrics.keys():
            all_metric_keys.add(k)

all_param_keys = sorted(list(all_param_keys))
all_metric_keys = sorted(list(all_metric_keys))

base_columns = [
    "experiment_id", "experiment_name", "run_id", "run_name",
    "status", "start_time", "end_time", "artifact_uri"
]
header = base_columns + ["param_" + k for k in all_param_keys] + ["metric_" + k for k in all_metric_keys]

if ARTIFACTS_DIR and not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

# Write CSV and download artifacts
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for exp, run in runs_info:
        info = run.info
        data = run.data
        run_id = info.run_id
        run_name = run.data.tags.get("mlflow.runName", "")
        status = info.status
        start_time = ""
        end_time = ""

        try:
            if info.start_time:
                start_time = datetime.fromtimestamp(info.start_time / 1000.0).isoformat()
            if info.end_time:
                end_time = datetime.fromtimestamp(info.end_time / 1000.0).isoformat()
        except Exception:
            start_time = ""
            end_time = ""

        artifact_uri = info.artifact_uri if hasattr(info, "artifact_uri") else ""

        row = [
            exp.experiment_id, exp.name, run_id, run_name,
            status, start_time, end_time, artifact_uri
        ]

        for pk in all_param_keys:
            row.append(data.params.get(pk, ""))

        for mk in all_metric_keys:
            row.append(str(data.metrics.get(mk, "")) if mk in data.metrics else "")

        writer.writerow(row)

        if ARTIFACTS_DIR:
            run_art_dir = os.path.join(ARTIFACTS_DIR, run_id)
            try:
                artifacts = client.list_artifacts(run_id, path="")
                for art in artifacts:
                    name = art.path
                    lower = name.lower()
                    if lower.endswith((".npy", ".npz", ".csv", ".png", ".jpg", ".jpeg")):
                        if not os.path.exists(run_art_dir):
                            os.makedirs(run_art_dir)
                        try:
                            client.download_artifacts(run_id, name, run_art_dir)
                        except Exception:
                            try:
                                children = client.list_artifacts(run_id, path=name)
                                for c in children:
                                    if not os.path.exists(run_art_dir):
                                        os.makedirs(run_art_dir)
                                    client.download_artifacts(run_id, c.path, run_art_dir)
                            except Exception:
                                pass
            except Exception:
                pass

if ARTIFACTS_DIR:
    mapping = {}
    for exp, run in runs_info:
        run_id = run.info.run_id
        run_dir = os.path.join(ARTIFACTS_DIR, run_id)
        if os.path.exists(run_dir):
            mapping[run_id] = run_dir

    with open("mlflow_artifact_map.json", "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)

print("Done. CSV written to: " + OUTPUT_CSV)
if ARTIFACTS_DIR:
    print("Artifacts saved (if found) under: " + ARTIFACTS_DIR)
    print("Artifact map: mlflow_artifact_map.json")