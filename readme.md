##  Demo Video





https://github.com/user-attachments/assets/262183f8-741d-4db9-9f7f-07f361842690




## Overview

This project performs **real-time bicep curl analysis** using pose keypoints from MediaPipe.
It evaluates form correctness, counts repetitions, and logs detailed rep-level performance metrics.

The system uses **rule-based biomechanics**, making it explainable and suitable for academic projects, research, and interviews.

---

## Features

### Real-time Bicep Curl Tracking

* Accurate elbow-angle detection
* Rep counting using angle thresholds + hysteresis
* Smooth angle curves using EMA smoothing
* Elbow stability detection
* Wrist–shoulder alignment tracking
* Torso tilt monitoring

### Rep-Level Quality Assessment

Each rep is scored as:

* GOOD FORM
* CAN IMPROVE
* BAD FORM

Based on biomechanical rules such as:

* Contraction depth (min elbow angle)
* Elbow drift away from torso
* Torso stability
* Wrist alignment

### Auto-Generated Feedback

Example console output:

```
Left REP 3 | quality: BAD FORM | tips: Keep elbow fixed; avoid drifting away from torso; Keep wrist aligned with shoulder
Right REP 5 | quality: CAN IMPROVE | tips: Keep wrist aligned with shoulder (avoid wrist drop/raise)
```

### MLflow Integration

The pipeline logs:

* Total reps (left/right)
* Average angles
* Good frame ratio
* Rep summary CSV
* Numpy angle arrays

---

## Project Structure

```
Form-Correctness-Detection-Using-Pose-Estimation/
│
├── examples_results/
├── outputs/
│   └── rep_summary.csv
│
├── src/
│   ├── bicep_curl_pipeline.py
│   ├── utils_math.py
│   ├── export_mlflow_summary.py
│   ├── mlruns/
│   └── mlflow_artifacts/
│── run_inference.py
├── requirements.txt
└── README.md
```

---

## Installation

### Step 1: Clone the repository

```bash
git clone <your_repo_url>
cd Form-Correctness-Detection-Using-Pose-Estimation
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Bicep Curl Pipeline

### Webcam (default):

```bash
python src/bicep_curl_pipeline.py --video 0
```

### Running inference wrapper:

```bash
python run_inference.py
```

---

## What the Pipeline Measures

### 1. Elbow Angle

Primary metric for contraction/extension.

### 2. Contraction Depth

How close the elbow angle gets to full curl.
Example rule:

* Angle < 30° → good depth
* Angle > 30° → shallow depth

### 3. Elbow Drift

How much the elbow moves away from the torso.

### 4. Wrist–Shoulder Alignment

Detects wrist drop or bending during curl.

### 5. Torso Stability

Checks if body sways to gain momentum.

---

## Example Console Output

```
Frame 742 | Reps: 3 | L_angle: 42.4 | R_angle: 46.6 | Form: CAN IMPROVE
Left REP 3 | quality: BAD FORM | Tips: Keep elbow fixed; avoid drifting from torso
Right REP 3 | quality: CAN IMPROVE | Tips: Keep wrist aligned with shoulder
```

---

## CSV Output (rep_summary.csv)

Each rep includes metrics:

| Field                   | Description                |
| ----------------------- | -------------------------- |
| rep_index               | Rep number                 |
| arm_side                | Left or right              |
| min_angle               | Deepest contraction        |
| max_angle               | Extension angle            |
| avg_angle               | Mean angle over rep        |
| max_elbow_drift         | Horizontal elbow shift     |
| max_torso_tilt          | Body stability             |
| max_wrist_shoulder_diff | Wrist alignment pressure   |
| quality_note            | good_depth / shallow_depth |
| flags                   | drift, tilt, wrist issues  |


## Acknowledgements

* MediaPipe Pose
* OpenCV
* NumPy
* MLflow


