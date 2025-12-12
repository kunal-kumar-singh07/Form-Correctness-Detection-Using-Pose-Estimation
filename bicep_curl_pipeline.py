
import cv2
import mediapipe as mp
import numpy as np
import mlflow
from utils_math import angle_3pts, EMASmoother

# Choose which side to use: "left" or "right"
ARM_SIDE = "left"  # change to "right" if you wanna use right arm

# MediaPipe landmark indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

# minimum visibility to accept a landmark
MIN_VISIBILITY = 0.45

# rep counting thresholds (degrees)
# When angle is below CONTRACTED_ANGLE we consider the arm to be "up"
# When angle is above EXTENDED_ANGLE we consider the arm to be "down"
CONTRACTED_ANGLE = 50.0
EXTENDED_ANGLE = 150.0

# some tolerance to avoid jitter (hysteresis)
HYSTERESIS = 8.0


def get_xy_vis(landmarks, idx, width, height):
    """
    Convert a mediapipe landmark to pixel coords and visibility.
    Returns (x:int, y:int, visibility:float or None)
    """
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
    """
    video_path: 0 for webcam or path to a video file.
    ema_alpha: smoothing parameter for the elbow angle.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video source: " + str(video_path))

    # pick indices depending on side
    if ARM_SIDE == "left":
        SHOULDER_IDX = LEFT_SHOULDER
        ELBOW_IDX = LEFT_ELBOW
        WRIST_IDX = LEFT_WRIST
    else:
        SHOULDER_IDX = RIGHT_SHOULDER
        ELBOW_IDX = RIGHT_ELBOW
        WRIST_IDX = RIGHT_WRIST

    smoother = EMASmoother(alpha=ema_alpha)

    # stats
    raw_angles = []
    smooth_angles = []
    good_frames = 0
    total_frames = 0

    # rep counting state machine
    # states: "idle" (waiting), "going_up" (contracting), "going_down" (extending)
    state = "idle"
    reps = 0
    last_angle = None

    # for logging an overall "quality" we can use good frames ratio
    mlflow.start_run(run_name="bicep_curl_counting_run")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

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

            # default feedback
            feedback_text = "No person / pose found"
            phase_text = "phase: unknown"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # draw full skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # get the three main points
                sx, sy, svis = get_xy_vis(landmarks, SHOULDER_IDX, width, height)
                ex, ey, evis = get_xy_vis(landmarks, ELBOW_IDX, width, height)
                wx, wy, wvis = get_xy_vis(landmarks, WRIST_IDX, width, height)

                # draw big dots for debugging
                try:
                    cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)   # shoulder = red
                    cv2.circle(frame, (ex, ey), 8, (0, 255, 255), -1) # elbow = yellow
                    cv2.circle(frame, (wx, wy), 8, (255, 0, 0), -1)   # wrist = blue
                except Exception:
                    # in case of weird coords, ignore drawing
                    pass

                # show visibility on frame
                vis_text = "vis S:" + "{:.2f}".format(svis if svis is not None else 0.0) + \
                           " E:" + "{:.2f}".format(evis if evis is not None else 0.0) + \
                           " W:" + "{:.2f}".format(wvis if wvis is not None else 0.0)
                cv2.putText(frame, vis_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # print debug info in terminal (helps when tuning)
                print("shoulder:", str(sx), str(sy), "vis:", str(svis))
                print("elbow:  ", str(ex), str(ey), "vis:", str(evis))
                print("wrist:  ", str(wx), str(wy), "vis:", str(wvis))

                # check visibility before using the points
                if (svis is not None and evis is not None and wvis is not None
                        and svis >= MIN_VISIBILITY and evis >= MIN_VISIBILITY and wvis >= MIN_VISIBILITY):

                    # compute raw and smoothed angle
                    raw_angle = angle_3pts((sx, sy), (ex, ey), (wx, wy))
                    raw_angles.append(raw_angle)
                    smooth_angle = smoother.update(raw_angle)
                    smooth_angles.append(smooth_angle)
                    total_frames += 1

                    # define whether current frame is "reasonably good"
                    # this rule is relaxed: we accept endpoints (contracted/extended) as OK
                    LOWER_OK = 20.0   # extremely low angle still considered okay (very contracted)
                    UPPER_OK = 170.0  # extremely high angle still okay (very extended)

                    if LOWER_OK <= smooth_angle <= UPPER_OK:
                        # frame is within overall plausible range
                        frame_good = True
                        good_frames += 1
                    else:
                        frame_good = False

                    # now the state machine for rep counting
                    # we use hysteresis to avoid flicker (small HYSTERESIS defined earlier)
                    contracted_threshold = CONTRACTED_ANGLE + HYSTERESIS
                    extended_threshold = EXTENDED_ANGLE - HYSTERESIS

                    # if angle is small enough -> we are contracting (going up)
                    if smooth_angle < contracted_threshold:
                        # entering or continuing the up phase
                        if state != "going_up":
                            # update state
                            state = "going_up"
                        phase_text = "phase: up"
                    # if angle is large enough -> we are extending (going down)
                    elif smooth_angle > extended_threshold:
                        if state == "going_up":
                            # we were going up and now returned down -> count a rep
                            reps += 1
                            # reset to idle briefly
                            state = "going_down"
                        else:
                            # either continue down or idle
                            state = "idle" if state == "idle" else state
                        phase_text = "phase: down"
                    else:
                        # mid-range between thresholds: the arm is moving
                        if state == "going_up":
                            phase_text = "phase: going up"
                        else:
                            phase_text = "phase: moving"

                    # feedback text: show angle, phase and reps
                    feedback_text = "angle: " + "{:.1f}".format(smooth_angle) + " deg" + "  | " + phase_text + "  | reps: " + str(reps)

                else:
                    # not enough confidence in landmarks, skip angle
                    feedback_text = "Landmarks low visibility - skip angle"
                    phase_text = "phase: unknown"

            # draw a small info box and the feedback text
            cv2.rectangle(frame, (8, 8), (860, 92), (0, 0, 0), -1)
            cv2.putText(frame, feedback_text, (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # also show the phase separately to make it clear
            cv2.putText(frame, phase_text, (16, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow("Bicep Curl Rep Counter - press q to quit", frame)

            # quit condition
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

    # final metrics and mlflow logging
    if total_frames > 0:
        good_ratio = float(good_frames) / float(total_frames)
        avg_angle = float(np.mean(smooth_angles)) if len(smooth_angles) > 0 else 0.0

        # print a short summary
        print("Total frames: " + str(total_frames))
        print("Good frames: " + str(good_frames))
        print("Good ratio: " + "{:.3f}".format(good_ratio))
        print("Total reps: " + str(reps))
        print("Average elbow angle: " + "{:.2f}".format(avg_angle))

        # log to mlflow
        mlflow.log_param("arm_side", ARM_SIDE)
        mlflow.log_param("video_source", str(video_path))
        mlflow.log_metric("total_frames", total_frames)
        mlflow.log_metric("good_frames", good_frames)
        mlflow.log_metric("good_ratio", good_ratio)
        mlflow.log_metric("total_reps", reps)
        mlflow.log_metric("avg_elbow_angle", avg_angle)

        # save and upload angles
        np.save("smooth_elbow_angles.npy", np.array(smooth_angles))
        mlflow.log_artifact("smooth_elbow_angles.npy")

    mlflow.end_run()


if __name__ == "__main__":
    run_pipeline(video_path=0)
