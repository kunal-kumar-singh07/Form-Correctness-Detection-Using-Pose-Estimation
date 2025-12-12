import csv
import os
import cv2
import mediapipe as mp
import numpy as np
import mlflow
from utils_math import angle_3pts, EMASmoother

# pick side: "left" or "right"
ARM_SIDE = "left"

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

MIN_VISIBILITY = 0.45

CONTRACTED_ANGLE = 50.0
EXTENDED_ANGLE = 150.0

HYSTERESIS = 8.0


def get_xy_vis(landmarks, idx, width, height):
    lm = landmarks[idx]
    x = int(lm.x * width)
    y = int(lm.y * height)
    vis = None
    if hasattr(lm, "visibility"):
        try:
            vis = float(lm.visibility)
        except Exception:
            vis = None
    return (x, y, vis)


def run_pipeline(video_path=0, ema_alpha=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video source: " + str(video_path))

    if ARM_SIDE == "left":
        SHOULDER_IDX = LEFT_SHOULDER
        ELBOW_IDX = LEFT_ELBOW
        WRIST_IDX = LEFT_WRIST
    else:
        SHOULDER_IDX = RIGHT_SHOULDER
        ELBOW_IDX = RIGHT_ELBOW
        WRIST_IDX = RIGHT_WRIST

    smoother = EMASmoother(alpha=ema_alpha)

    raw_angles = []
    smooth_angles = []
    good_frames = 0
    total_frames = 0

    state = "idle"
    reps = 0
    last_angle = None

    # per-rep tracking
    current_rep = None
    rep_records = []

    mlflow.start_run(run_name="bicep_curl_counting_run")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            feedback_text = "No person / pose found"
            phase_text = "phase: unknown"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                sx, sy, svis = get_xy_vis(landmarks, SHOULDER_IDX, width, height)
                ex, ey, evis = get_xy_vis(landmarks, ELBOW_IDX, width, height)
                wx, wy, wvis = get_xy_vis(landmarks, WRIST_IDX, width, height)

                try:
                    cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (ex, ey), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (wx, wy), 8, (255, 0, 0), -1)
                except Exception:
                    pass

                vis_text = "vis S:" + "{:.2f}".format(svis if svis is not None else 0.0) + \
                           " E:" + "{:.2f}".format(evis if evis is not None else 0.0) + \
                           " W:" + "{:.2f}".format(wvis if wvis is not None else 0.0)
                cv2.putText(frame, vis_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                print("shoulder:", str(sx), str(sy), "vis:", str(svis))
                print("elbow:  ", str(ex), str(ey), "vis:", str(evis))
                print("wrist:  ", str(wx), str(wy), "vis:", str(wvis))

                if (svis is not None and evis is not None and wvis is not None
                        and svis >= MIN_VISIBILITY and evis >= MIN_VISIBILITY and wvis >= MIN_VISIBILITY):

                    raw_angle = angle_3pts((sx, sy), (ex, ey), (wx, wy))
                    raw_angles.append(raw_angle)
                    smooth_angle = smoother.update(raw_angle)
                    smooth_angles.append(smooth_angle)
                    total_frames += 1

                    LOWER_OK = 20.0
                    UPPER_OK = 170.0

                    if LOWER_OK <= smooth_angle <= UPPER_OK:
                        frame_good = True
                        good_frames += 1
                    else:
                        frame_good = False

                    contracted_threshold = CONTRACTED_ANGLE + HYSTERESIS
                    extended_threshold = EXTENDED_ANGLE - HYSTERESIS

                    if smooth_angle < contracted_threshold:
                        if state != "going_up":
                            state = "going_up"
                        phase_text = "phase: up"
                    elif smooth_angle > extended_threshold:
                        # finalize rep if we were going_up
                        if state == "going_up":
                            state = "going_down"
                            # finalize current rep record
                            if current_rep is not None:
                                current_rep["end_frame"] = frame_idx
                                current_rep["duration_frames"] = current_rep["end_frame"] - current_rep["start_frame"]
                                angles = current_rep.get("angles", [])
                                if len(angles) > 0:
                                    current_rep["avg_angle"] = float(sum(angles) / len(angles))
                                    current_rep["min_angle"] = float(min(angles))
                                    current_rep["max_angle"] = float(max(angles))
                                else:
                                    current_rep["avg_angle"] = ""
                                    current_rep["min_angle"] = ""
                                    current_rep["max_angle"] = ""
                                # simple quality note
                                if current_rep.get("min_angle", 9999) < 30.0:
                                    current_rep["quality_note"] = "good_depth"
                                else:
                                    current_rep["quality_note"] = "shallow_depth"
                                current_rep["rep_index"] = reps + 1
                                rep_records.append(current_rep)
                                current_rep = None
                            reps += 1
                        else:
                            state = "idle" if state == "idle" else state
                        phase_text = "phase: down"
                    else:
                        if state == "going_up":
                            phase_text = "phase: going up"
                        else:
                            phase_text = "phase: moving"

                    # start a rep when contraction begins
                    if smooth_angle < CONTRACTED_ANGLE + HYSTERESIS and (current_rep is None):
                        current_rep = {
                            "start_frame": frame_idx,
                            "angles": [smooth_angle]
                        }

                    if current_rep is not None:
                        current_rep.setdefault("angles", []).append(smooth_angle)

                    feedback_text = "angle: " + "{:.1f}".format(smooth_angle) + " deg" + "  | " + phase_text + "  | reps: " + str(reps)

                else:
                    feedback_text = "Landmarks low visibility - skip angle"
                    phase_text = "phase: unknown"

            cv2.rectangle(frame, (8, 8), (860, 92), (0, 0, 0), -1)
            cv2.putText(frame, feedback_text, (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, phase_text, (16, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow("Bicep Curl Rep Counter - press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # save rep summary CSV
    if len(rep_records) > 0:
        csv_name = "rep_summary.csv"
        fieldnames = [
            "rep_index",
            "start_frame",
            "end_frame",
            "duration_frames",
            "avg_angle",
            "min_angle",
            "max_angle",
            "quality_note"
        ]
        try:
            with open(csv_name, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in rep_records:
                    row = {
                        "rep_index": r.get("rep_index", ""),
                        "start_frame": r.get("start_frame", ""),
                        "end_frame": r.get("end_frame", ""),
                        "duration_frames": r.get("duration_frames", ""),
                        "avg_angle": r.get("avg_angle", ""),
                        "min_angle": r.get("min_angle", ""),
                        "max_angle": r.get("max_angle", ""),
                        "quality_note": r.get("quality_note", "")
                    }
                    writer.writerow(row)
            mlflow.log_artifact(csv_name)
        except Exception:
            print("Could not write or log rep_summary.csv")

    if total_frames > 0:
        good_ratio = float(good_frames) / float(total_frames)
        avg_angle = float(np.mean(smooth_angles)) if len(smooth_angles) > 0 else 0.0

        print("Total frames: " + str(total_frames))
        print("Good frames: " + str(good_frames))
        print("Good ratio: " + "{:.3f}".format(good_ratio))
        print("Total reps: " + str(reps))
        print("Average elbow angle: " + "{:.2f}".format(avg_angle))

        mlflow.log_param("arm_side", ARM_SIDE)
        mlflow.log_param("video_source", str(video_path))
        mlflow.log_metric("total_frames", total_frames)
        mlflow.log_metric("good_frames", good_frames)
        mlflow.log_metric("good_ratio", good_ratio)
        mlflow.log_metric("total_reps", reps)
        mlflow.log_metric("avg_elbow_angle", avg_angle)

        np.save("smooth_elbow_angles.npy", np.array(smooth_angles))
        mlflow.log_artifact("smooth_elbow_angles.npy")

    mlflow.end_run()


if __name__ == "__main__":
    run_pipeline(video_path=0)
