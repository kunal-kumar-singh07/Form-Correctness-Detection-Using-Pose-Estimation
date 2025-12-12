import csv
import cv2
import mediapipe as mp
import numpy as np
import mlflow
from utils_math import angle_3pts, EMASmoother

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24

MIN_VISIBILITY = 0.45

CONTRACTED_ANGLE = 50.0
EXTENDED_ANGLE = 150.0

HYSTERESIS = 8.0

ELBOW_DRIFT_PIX_THRESH = 40
TORSO_TILT_DEG_THRESH = 15.0
WRIST_SHOULDER_PIX_THRESH = 35
GOOD_DEPTH_ANGLE = 30.0


def get_xy_vis(landmarks, idx, width, height):
    try:
        lm = landmarks[idx]
    except Exception:
        return 0, 0, None
    try:
        x = int(lm.x * width)
        y = int(lm.y * height)
    except Exception:
        return 0, 0, None
    vis = None
    if hasattr(lm, "visibility"):
        try:
            vis = float(lm.visibility)
        except Exception:
            vis = None
    return (x, y, vis)


def angle_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    ang = abs(np.degrees(np.arctan2(dx, dy)))
    return ang


def evaluate_bicep_form(avg_min_angle, elbow_drift_px, torso_tilt_deg, wrist_shoulder_px):
    suggestions = []
    issues = 0

    if avg_min_angle == "" or avg_min_angle is None:
        pass
    else:
        if avg_min_angle < GOOD_DEPTH_ANGLE:
            pass
        else:
            issues += 1
            suggestions.append("Bend elbow more (reach deeper contraction)")

    if elbow_drift_px and elbow_drift_px > ELBOW_DRIFT_PIX_THRESH:
        issues += 1
        suggestions.append("Keep elbow fixed; avoid drifting away from torso")

    if torso_tilt_deg and torso_tilt_deg > TORSO_TILT_DEG_THRESH:
        issues += 1
        suggestions.append("Keep your torso upright; avoid leaning/rocking")

    if wrist_shoulder_px and wrist_shoulder_px > WRIST_SHOULDER_PIX_THRESH:
        issues += 1
        suggestions.append("Keep wrist aligned with shoulder (avoid wrist drop/raise)")

    if issues == 0 and avg_min_angle != "" and avg_min_angle is not None and avg_min_angle < GOOD_DEPTH_ANGLE:
        label = "GOOD FORM"
    elif issues <= 1:
        label = "CAN IMPROVE"
    else:
        label = "BAD FORM"

    suggestions = suggestions[:2]

    return label, suggestions


def run_pipeline(video_path=0, ema_alpha=0.25, mirror=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video source: " + str(video_path))

    smoother_left = EMASmoother(alpha=ema_alpha)
    smoother_right = EMASmoother(alpha=ema_alpha)

    raw_angles_left = []
    smooth_angles_left = []
    raw_angles_right = []
    smooth_angles_right = []

    good_frames = 0
    total_frames = 0

    left_state = "idle"
    right_state = "idle"
    left_reps = 0
    right_reps = 0

    left_current_rep = None
    right_current_rep = None
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

            if mirror:
                frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            feedback_text = "No person / pose found"
            phase_text = "phase: unknown"
            form_label_left = ""
            form_suggestions_left = []
            form_label_right = ""
            form_suggestions_right = []

            if getattr(results, "pose_landmarks", None):
                landmarks = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                lsx, lsy, lsvis = get_xy_vis(landmarks, LEFT_SHOULDER, width, height)
                lex, ley, levis = get_xy_vis(landmarks, LEFT_ELBOW, width, height)
                lwx, lwy, lwvis = get_xy_vis(landmarks, LEFT_WRIST, width, height)
                lhx, lhy, lhvis = get_xy_vis(landmarks, LEFT_HIP, width, height)

                rsx, rsy, rsvis = get_xy_vis(landmarks, RIGHT_SHOULDER, width, height)
                rex, rey, revis = get_xy_vis(landmarks, RIGHT_ELBOW, width, height)
                rwx, rwy, rwvis = get_xy_vis(landmarks, RIGHT_WRIST, width, height)
                rhx, rhy, rhvis = get_xy_vis(landmarks, RIGHT_HIP, width, height)

                try:
                    cv2.circle(frame, (lsx, lsy), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (lex, ley), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (lwx, lwy), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (rsx, rsy), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (rex, rey), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (rwx, rwy), 6, (255, 0, 0), -1)
                except Exception:
                    pass

                vis_text = (
                    "L vis S:{:.2f} E:{:.2f} W:{:.2f} | R vis S:{:.2f} E:{:.2f} W:{:.2f}".format(
                        lsvis if lsvis is not None else 0.0,
                        levis if levis is not None else 0.0,
                        lwvis if lwvis is not None else 0.0,
                        rsvis if rsvis is not None else 0.0,
                        revis if revis is not None else 0.0,
                        rwvis if rwvis is not None else 0.0,
                    )
                )
                try:
                    cv2.putText(frame, vis_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception:
                    pass

                left_visible = (lsvis is not None and levis is not None and lwvis is not None and
                                lsvis >= MIN_VISIBILITY and levis >= MIN_VISIBILITY and lwvis >= MIN_VISIBILITY)
                if left_visible:
                    raw_angle_l = angle_3pts((lsx, lsy), (lex, ley), (lwx, lwy))
                    raw_angles_left.append(raw_angle_l)
                    smooth_angle_l = smoother_left.update(raw_angle_l)
                    smooth_angles_left.append(smooth_angle_l)

                    LOWER_OK = 20.0
                    UPPER_OK = 170.0
                    if LOWER_OK <= smooth_angle_l <= UPPER_OK:
                        good_frames += 1
                    total_frames += 1

                    contracted_threshold = CONTRACTED_ANGLE + HYSTERESIS
                    extended_threshold = EXTENDED_ANGLE - HYSTERESIS

                    elbow_drift_l = abs(lex - lsx)
                    torso_tilt_l = angle_vertical((lhx, lhy), (lsx, lsy))
                    wrist_shoulder_diff_l = abs(lwy - lsy)

                    min_angle_est = smooth_angle_l

                    if smooth_angle_l < contracted_threshold:
                        if left_state != "going_up":
                            left_state = "going_up"
                        left_phase = "up"
                    elif smooth_angle_l > extended_threshold:
                        if left_state == "going_up":
                            left_state = "going_down"
                            if left_current_rep is not None:
                                left_current_rep["end_frame"] = frame_idx
                                left_current_rep["duration_frames"] = left_current_rep["end_frame"] - left_current_rep["start_frame"]
                                angles = left_current_rep.get("angles", [])
                                if len(angles) > 0:
                                    left_current_rep["avg_angle"] = float(np.mean(angles))
                                    left_current_rep["min_angle"] = float(min(angles))
                                    left_current_rep["max_angle"] = float(max(angles))
                                else:
                                    left_current_rep["avg_angle"] = ""
                                    left_current_rep["min_angle"] = ""
                                    left_current_rep["max_angle"] = ""

                                drift_vals = left_current_rep.get("elbow_drift_vals", [])
                                tilt_vals = left_current_rep.get("torso_tilt_vals", [])
                                wrist_vals = left_current_rep.get("wrist_shoulder_vals", [])

                                left_current_rep["max_elbow_drift"] = max(drift_vals) if drift_vals else ""
                                left_current_rep["max_torso_tilt"] = max(tilt_vals) if tilt_vals else ""
                                left_current_rep["max_wrist_shoulder_diff"] = max(wrist_vals) if wrist_vals else ""

                                left_current_rep["elbow_drift_flag"] = 1 if drift_vals and max(drift_vals) > ELBOW_DRIFT_PIX_THRESH else 0
                                left_current_rep["torso_tilt_flag"] = 1 if tilt_vals and max(tilt_vals) > TORSO_TILT_DEG_THRESH else 0
                                left_current_rep["wrist_align_flag"] = 1 if wrist_vals and max(wrist_vals) > WRIST_SHOULDER_PIX_THRESH else 0

                                rep_min_angle = left_current_rep.get("min_angle", "")
                                rep_drift_px = left_current_rep.get("max_elbow_drift", 0)
                                rep_tilt_deg = left_current_rep.get("max_torso_tilt", 0)
                                rep_wrist_px = left_current_rep.get("max_wrist_shoulder_diff", 0)

                                label, tips = evaluate_bicep_form(rep_min_angle, rep_drift_px, rep_tilt_deg, rep_wrist_px)
                                left_current_rep["quality_note"] = label
                                left_current_rep["suggestions"] = "; ".join(tips) if tips else "Good"

                                left_current_rep["rep_index"] = left_reps + 1
                                left_current_rep["arm_side"] = "left"

                                rep_records.append(left_current_rep)
                                mlflow.log_param(f"left_rep_{left_current_rep['rep_index']}_quality", label)
                                mlflow.log_param(f"left_rep_{left_current_rep['rep_index']}_tips", left_current_rep["suggestions"])

                                print(f"Left REP {left_current_rep['rep_index']} | quality: {label} | tips: {left_current_rep['suggestions']}")

                                left_current_rep = None
                            left_reps += 1
                        else:
                            left_state = "idle" if left_state == "idle" else left_state
                        left_phase = "down"
                    else:
                        left_phase = "going up" if left_state == "going_up" else "moving"

                    if smooth_angle_l < CONTRACTED_ANGLE + HYSTERESIS and (left_current_rep is None):
                        left_current_rep = {
                            "start_frame": frame_idx,
                            "angles": [smooth_angle_l],
                            "elbow_drift_vals": [elbow_drift_l],
                            "torso_tilt_vals": [torso_tilt_l],
                            "wrist_shoulder_vals": [wrist_shoulder_diff_l],
                        }
                    elif left_current_rep is not None:
                        left_current_rep.setdefault("angles", []).append(smooth_angle_l)
                        left_current_rep.setdefault("elbow_drift_vals", []).append(elbow_drift_l)
                        left_current_rep.setdefault("torso_tilt_vals", []).append(torso_tilt_l)
                        left_current_rep.setdefault("wrist_shoulder_vals", []).append(wrist_shoulder_diff_l)

                    inst_label, inst_tips = evaluate_bicep_form(smooth_angle_l, elbow_drift_l, torso_tilt_l, wrist_shoulder_diff_l)
                    form_label_left = inst_label
                    form_suggestions_left = inst_tips

                    feedback_text = f"L angle: {smooth_angle_l:.1f} deg | Lphase: {left_phase} | Lreps: {left_reps}"
                else:
                    feedback_text = "Landmarks low visibility - skip left"

                right_visible = (rsvis is not None and revis is not None and rwvis is not None and
                                 rsvis >= MIN_VISIBILITY and revis >= MIN_VISIBILITY and rwvis >= MIN_VISIBILITY)
                if right_visible:
                    raw_angle_r = angle_3pts((rsx, rsy), (rex, rey), (rwx, rwy))
                    raw_angles_right.append(raw_angle_r)
                    smooth_angle_r = smoother_right.update(raw_angle_r)
                    smooth_angles_right.append(smooth_angle_r)

                    LOWER_OK = 20.0
                    UPPER_OK = 170.0
                    if LOWER_OK <= smooth_angle_r <= UPPER_OK:
                        good_frames += 1
                    total_frames += 1

                    contracted_threshold = CONTRACTED_ANGLE + HYSTERESIS
                    extended_threshold = EXTENDED_ANGLE - HYSTERESIS

                    elbow_drift_r = abs(rex - rsx)
                    torso_tilt_r = angle_vertical((rhx, rhy), (rsx, rsy))
                    wrist_shoulder_diff_r = abs(rwy - rsy)

                    if smooth_angle_r < contracted_threshold:
                        if right_state != "going_up":
                            right_state = "going_up"
                        right_phase = "up"
                    elif smooth_angle_r > extended_threshold:
                        if right_state == "going_up":
                            right_state = "going_down"
                            if right_current_rep is not None:
                                right_current_rep["end_frame"] = frame_idx
                                right_current_rep["duration_frames"] = right_current_rep["end_frame"] - right_current_rep["start_frame"]
                                angles = right_current_rep.get("angles", [])
                                if len(angles) > 0:
                                    right_current_rep["avg_angle"] = float(np.mean(angles))
                                    right_current_rep["min_angle"] = float(min(angles))
                                    right_current_rep["max_angle"] = float(max(angles))
                                else:
                                    right_current_rep["avg_angle"] = ""
                                    right_current_rep["min_angle"] = ""
                                    right_current_rep["max_angle"] = ""

                                drift_vals = right_current_rep.get("elbow_drift_vals", [])
                                tilt_vals = right_current_rep.get("torso_tilt_vals", [])
                                wrist_vals = right_current_rep.get("wrist_shoulder_vals", [])

                                right_current_rep["max_elbow_drift"] = max(drift_vals) if drift_vals else ""
                                right_current_rep["max_torso_tilt"] = max(tilt_vals) if tilt_vals else ""
                                right_current_rep["max_wrist_shoulder_diff"] = max(wrist_vals) if wrist_vals else ""

                                right_current_rep["elbow_drift_flag"] = 1 if drift_vals and max(drift_vals) > ELBOW_DRIFT_PIX_THRESH else 0
                                right_current_rep["torso_tilt_flag"] = 1 if tilt_vals and max(tilt_vals) > TORSO_TILT_DEG_THRESH else 0
                                right_current_rep["wrist_align_flag"] = 1 if wrist_vals and max(wrist_vals) > WRIST_SHOULDER_PIX_THRESH else 0

                                rep_min_angle = right_current_rep.get("min_angle", "")
                                rep_drift_px = right_current_rep.get("max_elbow_drift", 0)
                                rep_tilt_deg = right_current_rep.get("max_torso_tilt", 0)
                                rep_wrist_px = right_current_rep.get("max_wrist_shoulder_diff", 0)

                                label, tips = evaluate_bicep_form(rep_min_angle, rep_drift_px, rep_tilt_deg, rep_wrist_px)
                                right_current_rep["quality_note"] = label
                                right_current_rep["suggestions"] = "; ".join(tips) if tips else "Good"

                                right_current_rep["rep_index"] = right_reps + 1
                                right_current_rep["arm_side"] = "right"

                                rep_records.append(right_current_rep)
                                mlflow.log_param(f"right_rep_{right_current_rep['rep_index']}_quality", label)
                                mlflow.log_param(f"right_rep_{right_current_rep['rep_index']}_tips", right_current_rep["suggestions"])
                                print(f"Right REP {right_current_rep['rep_index']} | quality: {label} | tips: {right_current_rep['suggestions']}")

                                right_current_rep = None
                            right_reps += 1
                        else:
                            right_state = "idle" if right_state == "idle" else right_state
                        right_phase = "down"
                    else:
                        right_phase = "going up" if right_state == "going_up" else "moving"

                    if smooth_angle_r < CONTRACTED_ANGLE + HYSTERESIS and (right_current_rep is None):
                        right_current_rep = {
                            "start_frame": frame_idx,
                            "angles": [smooth_angle_r],
                            "elbow_drift_vals": [elbow_drift_r],
                            "torso_tilt_vals": [torso_tilt_r],
                            "wrist_shoulder_vals": [wrist_shoulder_diff_r],
                        }
                    elif right_current_rep is not None:
                        right_current_rep.setdefault("angles", []).append(smooth_angle_r)
                        right_current_rep.setdefault("elbow_drift_vals", []).append(elbow_drift_r)
                        right_current_rep.setdefault("torso_tilt_vals", []).append(torso_tilt_r)
                        right_current_rep.setdefault("wrist_shoulder_vals", []).append(wrist_shoulder_diff_r)

                    inst_label_r, inst_tips_r = evaluate_bicep_form(smooth_angle_r, elbow_drift_r, torso_tilt_r, wrist_shoulder_diff_r)
                    form_label_right = inst_label_r
                    form_suggestions_right = inst_tips_r

                    feedback_text = feedback_text + " | " + f"R angle: {smooth_angle_r:.1f} deg | Rphase: {right_phase} | Rreps: {right_reps}"
                else:
                    feedback_text = feedback_text + " | Landmarks low visibility - skip right"

            font = cv2.FONT_HERSHEY_SIMPLEX
            base_scale = 0.6
            thickness1 = 2
            thickness2 = 1

            (text_w1, text_h1), baseline1 = cv2.getTextSize(feedback_text, font, base_scale, thickness1)
            (text_w2, text_h2), baseline2 = cv2.getTextSize(phase_text, font, base_scale, thickness2)
            max_text_w = max(text_w1, text_w2)
            max_allowed_w = max(16, width - 32)

            scale = base_scale
            if max_text_w > max_allowed_w:
                scale = base_scale * (max_allowed_w / float(max_text_w))
                (text_w1, text_h1), baseline1 = cv2.getTextSize(feedback_text, font, scale, thickness1)
                (text_w2, text_h2), baseline2 = cv2.getTextSize(phase_text, font, scale, thickness2)

            pad_x = 12
            pad_y = 8
            box_w = min(width - 16, max(text_w1, text_w2) + 2 * pad_x)
            box_h = text_h1 + text_h2 + pad_y * 4 + 40

            box_x1 = 8
            box_y1 = 8
            box_x2 = box_x1 + int(box_w)
            box_y2 = box_y1 + int(box_h)

            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

            line1_y = box_y1 + pad_y + text_h1
            line2_y = line1_y + pad_y + text_h2

            cv2.putText(frame, feedback_text, (box_x1 + pad_x, line1_y),
                        font, scale, (0, 255, 0), thickness1, lineType=cv2.LINE_AA)
            cv2.putText(frame, phase_text, (box_x1 + pad_x, line2_y),
                        font, scale, (200, 200, 0), thickness2, lineType=cv2.LINE_AA)

            sug_y = line2_y + pad_y + 8
            if form_label_left:
                cv2.putText(frame, f"Lform: {form_label_left}", (box_x1 + pad_x, sug_y),
                            font, 0.55, (0, 255, 0) if form_label_left == "GOOD FORM" else (0, 215, 255) if form_label_left == "CAN IMPROVE" else (0, 128, 255),
                            1, cv2.LINE_AA)
                sug_y += 20
                if form_suggestions_left:
                    cv2.putText(frame, "L tip: " + ("; ".join(form_suggestions_left)), (box_x1 + pad_x, sug_y),
                                font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                    sug_y += 20

            if form_label_right:
                cv2.putText(frame, f"Rform: {form_label_right}", (box_x1 + pad_x, sug_y),
                            font, 0.55, (0, 255, 0) if form_label_right == "GOOD FORM" else (0, 215, 255) if form_label_right == "CAN IMPROVE" else (0, 128, 255),
                            1, cv2.LINE_AA)
                sug_y += 20
                if form_suggestions_right:
                    cv2.putText(frame, "R tip: " + ("; ".join(form_suggestions_right)), (box_x1 + pad_x, sug_y),
                                font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                    sug_y += 20

            cv2.imshow("Bicep Curl Rep Counter - press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(rep_records) > 0:
        csv_name = "rep_summary.csv"
        fieldnames = [
            "rep_index",
            "arm_side",
            "start_frame",
            "end_frame",
            "duration_frames",
            "avg_angle",
            "min_angle",
            "max_angle",
            "max_elbow_drift",
            "max_torso_tilt",
            "max_wrist_shoulder_diff",
            "elbow_drift_flag",
            "torso_tilt_flag",
            "wrist_align_flag",
            "quality_note",
            "suggestions"
        ]
        try:
            with open(csv_name, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in rep_records:
                    row = {
                        "rep_index": r.get("rep_index", ""),
                        "arm_side": r.get("arm_side", ""),
                        "start_frame": r.get("start_frame", ""),
                        "end_frame": r.get("end_frame", ""),
                        "duration_frames": r.get("duration_frames", ""),
                        "avg_angle": r.get("avg_angle", ""),
                        "min_angle": r.get("min_angle", ""),
                        "max_angle": r.get("max_angle", ""),
                        "max_elbow_drift": r.get("max_elbow_drift", ""),
                        "max_torso_tilt": r.get("max_torso_tilt", ""),
                        "max_wrist_shoulder_diff": r.get("max_wrist_shoulder_diff", ""),
                        "elbow_drift_flag": r.get("elbow_drift_flag", ""),
                        "torso_tilt_flag": r.get("torso_tilt_flag", ""),
                        "wrist_align_flag": r.get("wrist_align_flag", ""),
                        "quality_note": r.get("quality_note", ""),
                        "suggestions": r.get("suggestions", "")
                    }
                    writer.writerow(row)
            mlflow.log_artifact(csv_name)
        except Exception:
            print("Could not write or log rep_summary.csv")

    if total_frames > 0:
        good_ratio = float(good_frames) / float(total_frames) if total_frames else 0.0
        avg_angle_left = float(np.mean(smooth_angles_left)) if len(smooth_angles_left) > 0 else 0.0
        avg_angle_right = float(np.mean(smooth_angles_right)) if len(smooth_angles_right) > 0 else 0.0

        print("Total frames: " + str(total_frames))
        print("Good frames: " + str(good_frames))
        print("Good ratio: " + "{:.3f}".format(good_ratio))
        print("Total reps left: " + str(left_reps))
        print("Total reps right: " + str(right_reps))
        print("Average elbow angle left: " + "{:.2f}".format(avg_angle_left))
        print("Average elbow angle right: " + "{:.2f}".format(avg_angle_right))

        mlflow.log_param("video_source", str(video_path))
        mlflow.log_metric("total_frames", total_frames)
        mlflow.log_metric("good_frames", good_frames)
        mlflow.log_metric("good_ratio", good_ratio)
        mlflow.log_metric("total_reps_left", left_reps)
        mlflow.log_metric("total_reps_right", right_reps)
        mlflow.log_metric("avg_elbow_angle_left", avg_angle_left)
        mlflow.log_metric("avg_elbow_angle_right", avg_angle_right)

        np.save("smooth_elbow_angles_left.npy", np.array(smooth_angles_left))
        np.save("smooth_elbow_angles_right.npy", np.array(smooth_angles_right))
        mlflow.log_artifact("smooth_elbow_angles_left.npy")
        mlflow.log_artifact("smooth_elbow_angles_right.npy")

    mlflow.end_run()


if __name__ == "__main__":
    run_pipeline(video_path=0, ema_alpha=0.25, mirror=False)