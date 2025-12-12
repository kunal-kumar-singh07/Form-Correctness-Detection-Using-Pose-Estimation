
import cv2
import mediapipe as mp
import numpy as np
import mlflow
from utils_math import angle_3pts, EMASmoother


ARM_SIDE = "left"

# MediaPipe indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

# minimum visibility to accept a landmark
MIN_VISIBILITY = 0.45


def get_xy_vis(landmarks, idx, width, height):

    lm = landmarks[idx]
    x = int(lm.x * width)
    y = int(lm.y * height)
    vis = None
    # some mediapipe builds might not include 'visibility', but usually do
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

    # pick indices depending on side
    if ARM_SIDE == "left":
        SHOULDER_IDX = LEFT_SHOULDER
        ELBOW_IDX = LEFT_ELBOW
        WRIST_IDX = LEFT_WRIST
    else:
        SHOULDER_IDX = RIGHT_SHOULDER
        ELBOW_IDX = RIGHT_ELBOW
        WRIST_IDX = RIGHT_WRIST

    # smoother for elbow angle
    smoother = EMASmoother(alpha=ema_alpha)

    # stats
    raw_angles = []
    smooth_angles = []
    good_frames = 0
    total_frames = 0

    reps = 0
    direction = None  # 'up' or 'down'
    last_angle = None

    # start mlflow run
    mlflow.start_run(run_name="bicep_curl_simple_run")

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
                # no more frames
                break

            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            feedback_text = "No person / pose found"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # draw whole skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # get pixel coords and visibility for the three points
                sx, sy, svis = get_xy_vis(landmarks, SHOULDER_IDX, width, height)
                ex, ey, evis = get_xy_vis(landmarks, ELBOW_IDX, width, height)
                wx, wy, wvis = get_xy_vis(landmarks, WRIST_IDX, width, height)

                # draw big dots for the 3 points so we can easily see them
                # red = shoulder, yellow = elbow, blue = wrist
                try:
                    cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)   # red
                    cv2.circle(frame, (ex, ey), 8, (0, 255, 255), -1) # yellow
                    cv2.circle(frame, (wx, wy), 8, (255, 0, 0), -1)   # blue
                except Exception:
                    # in case coordinates are weird, skip drawing
                    pass

                # show raw visibility numbers on the frame
                vis_text = "vis S:" + "{:.2f}".format(svis if svis is not None else 0.0) + \
                           " E:" + "{:.2f}".format(evis if evis is not None else 0.0) + \
                           " W:" + "{:.2f}".format(wvis if wvis is not None else 0.0)
                cv2.putText(frame, vis_text, (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # print the values in the terminal for debugging
                # this helps us see if points are on the face or arm
                print("shoulder:", str(sx), str(sy), "vis:", str(svis))
                print("elbow:  ", str(ex), str(ey), "vis:", str(evis))
                print("wrist:  ", str(wx), str(wy), "vis:", str(wvis))

                # only compute angle if visibility is acceptable for all three
                if (svis is not None and evis is not None and wvis is not None
                        and svis >= MIN_VISIBILITY and evis >= MIN_VISIBILITY and wvis >= MIN_VISIBILITY):

                    # compute raw angle (in degrees)
                    raw_angle = angle_3pts((sx, sy), (ex, ey), (wx, wy))
                    raw_angles.append(raw_angle)

                    # smooth it
                    smooth_angle = smoother.update(raw_angle)
                    smooth_angles.append(smooth_angle)

                    total_frames += 1

                    # basic rule: accept angle if in range [40, 160]
                    if 40 <= smooth_angle <= 160:
                        feedback_text = "Good form | angle: " + "{:.1f}".format(smooth_angle) + " deg"
                        good_frames += 1
                    else:
                        feedback_text = "Check form | angle: " + "{:.1f}".format(smooth_angle) + " deg"

                    # very simple rep counting
                    if last_angle is not None:
                        # arm moved up
                        if (direction is None or direction == "down") and smooth_angle < 60 and last_angle >= 60:
                            direction = "up"
                        # arm moved down and we count the rep
                        elif direction == "up" and smooth_angle > 150 and last_angle <= 150:
                            direction = "down"
                            reps += 1
                            feedback_text = feedback_text + "  | Rep " + str(reps) + " completed"

                    last_angle = smooth_angle

                else:
                    # not enough confidence in landmarks
                    feedback_text = "Landmarks low visibility - skipping angle"

            # overlay the feedback text
            cv2.rectangle(frame, (8, 8), (720, 72), (0, 0, 0), -1)
            cv2.putText(frame, feedback_text, (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

            # show the frame
            cv2.imshow("Bicep Curl Debug - press q to quit", frame)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

    # final metrics and mlflow logging
    if total_frames > 0:
        good_ratio = float(good_frames) / float(total_frames)
        avg_angle = float(np.mean(smooth_angles)) if len(smooth_angles) > 0 else 0.0

        # print summary
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

        # save angle series and upload as artifact
        np.save("smooth_elbow_angles.npy", np.array(smooth_angles))
        mlflow.log_artifact("smooth_elbow_angles.npy")

    mlflow.end_run()


if __name__ == "__main__":
    # default: webcam (0)
    run_pipeline(video_path=0)
