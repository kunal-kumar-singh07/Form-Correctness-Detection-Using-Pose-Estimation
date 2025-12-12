import os
import subprocess
import argparse
import sys

def run_cmd_inherit_io(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def find_pipeline_script():
    candidate = os.path.join("src", "bicep_curl_pipeline.py")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    for root, _, files in os.walk("."):
        if "bicep_curl_pipeline.py" in files:
            return os.path.abspath(os.path.join(root, "bicep_curl_pipeline.py"))
    return None

def find_summary_script():
    candidate = os.path.join("src", "export_mlflow_summary.py")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    for root, _, files in os.walk("."):
        if "export_mlflow_summary.py" in files:
            return os.path.abspath(os.path.join(root, "export_mlflow_summary.py"))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="0", help="camera index (0) or video path")
    parser.add_argument("--outdir", type=str, default="outputs/run1")
    parser.add_argument("--no-summary", action="store_true", help="skip mlflow summary export")
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    pipeline_script = find_pipeline_script()
    if not pipeline_script:
        print("ERROR: cannot find src/bicep_curl_pipeline.py")
        sys.exit(1)

    project_root = os.path.abspath(".")

    video_arg = args.video
    if video_arg == "0":
        video_arg = "0"
    cmd = [sys.executable, pipeline_script, "--video", video_arg, "--outdir", outdir]

    print("Running bicep curl pipeline...")
    run_cmd_inherit_io(cmd, cwd=project_root)

    if not args.no_summary:
        summary_script = find_summary_script()
        if summary_script:
            print("Exporting MLflow summary...")
            run_cmd_inherit_io([sys.executable, summary_script, "--outdir", outdir], cwd=project_root)

if __name__ == "__main__":
    main()