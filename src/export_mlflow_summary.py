# export_mlflow_summary.py
# Export MLflow runs into one CSV summary and optionally download artifacts.
# Works with different mlflow versions (no dependency on client.list_experiments()).

import os
import csv
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# ---------- CONFIG ----------
OUTPUT_CSV = "mlflow_summary.csv"
ARTIFACTS_DIR = "mlflow_artifacts"   # set to None to skip downloading artifacts
MLFLOW_TRACKING_URI = None           # set if using remote tracking server
# ----------------------------

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

# try to get all runs in a safe way
# mlflow.search_runs returns a DataFrame; we use it to discover run IDs
try:
    runs_df = mlflow.search_runs()
except Exception:
    # fallback: try with explicit experiment_ids if search_runs has different signature
    runs_df = mlflow.search_runs(experiment_ids=None)

run_rows = []
if runs_df is not None and len(runs_df) > 0:
    for idx, row in runs_df.iterrows():
        run_rows.append(row)

# if we found no runs with search_runs, try scanning mlruns folder
if len(run_rows) == 0:
    mlruns_path = mlflow.get_tracking_uri()
    if mlruns_path.startswith("file://"):
        mlruns_path = mlruns_path[len("file://"):]
    if not mlruns_path:
        mlruns_path = "mlruns"
    if os.path.exists(mlruns_path):
        # collect runs by directory structure mlruns/<exp_id>/<run_id>
        for exp_id in os.listdir(mlruns_path):
            exp_dir = os.path.join(mlruns_path, exp_id)
            if os.path.isdir(exp_dir):
                for run_id in os.listdir(exp_dir):
                    run_file = os.path.join(exp_dir, run_id)
                    # try to get run via client
                    try:
                        run = client.get_run(run_id)
                        run_rows.append(run)
                    except Exception:
                        # skip if cannot fetch
                        pass

# gather all param/metric names
all_param_keys = set()
all_metric_keys = set()
runs_info = []  # will contain Mlflow Run objects or dict-like rows

# if run_rows are pandas Series (from search_runs), convert to run list by fetching run objects
if len(run_rows) > 0:
    # detect type
    first = run_rows[0]
    # pandas Series (search_runs)
    try:
        import pandas as pd  # pragma: no cover
        if isinstance(first, pd.Series):
            # run ids are in 'run_id' column
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
            # maybe already Run objects
            for item in run_rows:
                if hasattr(item, "info") and hasattr(item, "data"):
                    runs_info.append(item)
                    for k in item.data.params.keys():
                        all_param_keys.add(k)
                    for k in item.data.metrics.keys():
                        all_metric_keys.add(k)
                else:
                    # unknown type: try to extract run_id and fetch
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
        # no pandas installed or unexpected types; try to use client.list_experiments/get_run if available
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

# sort keys for nicer CSV
all_param_keys = sorted(list(all_param_keys))
all_metric_keys = sorted(list(all_metric_keys))

# prepare CSV header
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
            # if item is not a Run object, skip
            continue

        run_id = info.run_id
        # get run name from tags if present
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
            "" ,  # experiment_name left empty (can be fetched with client.get_experiment but not required)
            run_id,
            run_name,
            status,
            start_time,
            end_time,
            artifact_uri
        ]

        # add params
        for pk in all_param_keys:
            row.append(data.params.get(pk, "") if hasattr(data, "params") else "")

        # add metrics (last value)
        for mk in all_metric_keys:
            if hasattr(data, "metrics") and mk in data.metrics:
                row.append(str(data.metrics.get(mk, "")))
            else:
                row.append("")

        writer.writerow(row)

        # download small useful artifacts for evaluator
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
                            # try to download children if it is a folder
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

# write artifact map
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
