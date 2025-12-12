import os
import csv
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

OUTPUT_CSV = "mlflow_summary.csv"
ARTIFACTS_DIR = "mlflow_artifacts"
MLFLOW_TRACKING_URI = None

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

try:
    runs_df = mlflow.search_runs()
except Exception:
    runs_df = mlflow.search_runs(experiment_ids=None)

run_rows = []
if runs_df is not None and len(runs_df) > 0:
    for idx, row in runs_df.iterrows():
        run_rows.append(row)

if len(run_rows) == 0:
    mlruns_path = mlflow.get_tracking_uri()
    if mlruns_path.startswith("file://"):
        mlruns_path = mlruns_path[len("file://"):]
    if not mlruns_path:
        mlruns_path = "mlruns"
    if os.path.exists(mlruns_path):
        for exp_id in os.listdir(mlruns_path):
            exp_dir = os.path.join(mlruns_path, exp_id)
            if os.path.isdir(exp_dir):
                for run_id in os.listdir(exp_dir):
                    run_file = os.path.join(exp_dir, run_id)
                    try:
                        run = client.get_run(run_id)
                        run_rows.append(run)
                    except Exception:
                        pass

all_param_keys = set()
all_metric_keys = set()
runs_info = []

if len(run_rows) > 0:
    first = run_rows[0]
    try:
        import pandas as pd
        if isinstance(first, pd.Series):
            for row in run_rows:
                run_id = row["run_id"]
                try:
                    run = client.get_run(run_id)
                    runs_info.append(run)
                    for k in run.data.params.keys():
                        all_param_keys.add(k)
                    for k in run.data.metrics.keys():
                        all_metric_keys.add(k)
                except Exception:
                    pass
        else:
            for item in run_rows:
                if hasattr(item, "info") and hasattr(item, "data"):
                    runs_info.append(item)
                    for k in item.data.params.keys():
                        all_param_keys.add(k)
                    for k in item.data.metrics.keys():
                        all_metric_keys.add(k)
                else:
                    try:
                        run_id = item.get("run_id", None)
                        if run_id:
                            run = client.get_run(run_id)
                            runs_info.append(run)
                            for k in run.data.params.keys():
                                all_param_keys.add(k)
                            for k in run.data.metrics.keys():
                                all_metric_keys.add(k)
                    except Exception:
                        pass
    except Exception:
        for item in run_rows:
            try:
                run_id = getattr(item, "run_id", None) or item.get("run_id", None)
                if run_id:
                    run = client.get_run(run_id)
                    runs_info.append(run)
                    for k in run.data.params.keys():
                        all_param_keys.add(k)
                    for k in run.data.metrics.keys():
                        all_metric_keys.add(k)
            except Exception:
                pass

all_param_keys = sorted(list(all_param_keys))
all_metric_keys = sorted(list(all_metric_keys))

base_columns = [
    "experiment_id",
    "experiment_name",
    "run_id",
    "run_name",
    "status",
    "start_time",
    "end_time",
    "artifact_uri"
]
header = base_columns + ["param_" + k for k in all_param_keys] + ["metric_" + k for k in all_metric_keys]

if ARTIFACTS_DIR:
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for run in runs_info:
        try:
            info = run.info
            data = run.data
        except Exception:
            continue

        run_id = info.run_id
        run_name = data.tags.get("mlflow.runName", "") if hasattr(data, "tags") else ""
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

        artifact_uri = getattr(info, "artifact_uri", "") or ""

        row = [
            getattr(run, "info").experiment_id if hasattr(run, "info") else "",
            "",
            run_id,
            run_name,
            status,
            start_time,
            end_time,
            artifact_uri
        ]

        for pk in all_param_keys:
            row.append(data.params.get(pk, "") if hasattr(data, "params") else "")

        for mk in all_metric_keys:
            if hasattr(data, "metrics") and mk in data.metrics:
                row.append(str(data.metrics.get(mk, "")))
            else:
                row.append("")

        writer.writerow(row)

        if ARTIFACTS_DIR:
            run_art_dir = os.path.join(ARTIFACTS_DIR, run_id)
            try:
                artifacts = client.list_artifacts(run_id, path="")
                for art in artifacts:
                    name = art.path
                    lower = name.lower()
                    if lower.endswith(".npy") or lower.endswith(".csv") or lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
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
    for run in runs_info:
        try:
            run_id = run.info.run_id
            run_dir = os.path.join(ARTIFACTS_DIR, run_id)
            if os.path.exists(run_dir):
                mapping[run_id] = run_dir
        except Exception:
            pass
    with open("mlflow_artifact_map.json", "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)

print("Done. CSV written to: " + OUTPUT_CSV)
if ARTIFACTS_DIR:
    print("Artifacts (if any) downloaded under: " + ARTIFACTS_DIR)