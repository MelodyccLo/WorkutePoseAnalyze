import cv2
import mediapipe as mp
import numpy as np
import math
import time 

# ================================================
#           CONFIGURATION PARAMETERS
# ================================================
# --- 1. Set the path to your video file ---
VIDEO_FILE_PATH = "test1.mov"

# --- 2. Set the desired width of the suggested angle range ---
#      (e.g., 20 means the range will be 20 degrees wide, like 80-100 or 0-20)
ANGLE_RANGE_WIDTH = 20
# ================================================


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# --- MODIFIED: The function now uses the global ANGLE_RANGE_WIDTH ---
def calculate_angle_range(angle, delta=10, round_to=5):
    """
    Calculates a clean, rounded range around an angle.
    """
    raw_lower_bound = 0
    raw_upper_bound = 0

    if (angle - delta) < 0:
        raw_lower_bound = 0
        raw_upper_bound = ANGLE_RANGE_WIDTH # Uses the global config variable
    elif (angle + delta) > 180:
        raw_lower_bound = 180 - ANGLE_RANGE_WIDTH # Uses the global config variable
        raw_upper_bound = 180
    else:
        raw_lower_bound = angle - delta
        raw_upper_bound = angle + delta

    final_lower_bound = round(raw_lower_bound / round_to) * round_to
    final_upper_bound = round(raw_upper_bound / round_to) * round_to

    return (int(final_lower_bound), int(final_upper_bound))

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points in a 0-180 degree range.
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_and_print_summary(all_data):
    """
    Calculates and prints the average angles for each category.
    """
    if not all_data:
        print("No data was categorized. Exiting.")
        return

    print("\n\n" + "="*50)
    print("           FINAL WORKOUT SUMMARY")
    print("="*50)

    sorted_categories = sorted(all_data.keys(), key=int)

    for category in sorted_categories:
        data_list = all_data[category]
        num_entries = len(data_list)
        
        if num_entries == 0:
            continue
        joint_names = data_list[0].keys()
        
        joint_sums = {joint: 0 for joint in joint_names}

        for data_snapshot in data_list:
            for joint, angle in data_snapshot.items():
                joint_sums[joint] += angle
        
        joint_averages = {joint: total / num_entries for joint, total in joint_sums.items()}

        print(f"\n--- Summary for Category: {category} ({num_entries} entries) ---")
        print(f"{'Angle:':<15} {'Average':>10} {'Min':>5} {'Max':>5}")
        print("-" * 38)
        
        for joint in joint_names:
            avg_angle = joint_averages[joint]
            avg_range = calculate_angle_range(avg_angle)
            print(f"{joint + ':':<15} {int(avg_angle):>10} {avg_range[0]:>5} {avg_range[1]:>5}")
        print("-" * 38)


def analyze_workout_video(video_path):
    """
    Analyzes a workout video, logs data, allows categorization, and provides a final summary.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    paused = False
    angles_data = {}
    all_workout_data = {}

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        if paused and results.pose_landmarks:
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (320, 380), (66, 117, 245), -1)
            alpha = 0.7
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            font, font_scale, font_color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            x_col_joint, x_col_orig, x_col_min, x_col_max = 20, 170, 220, 270
            y_pos = 40

            cv2.putText(image, "Angle:", (x_col_joint, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Orig", (x_col_orig, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Min", (x_col_min, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Max", (x_col_max, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            y_pos += 25
            cv2.putText(image, "-" * 19, (20, y_pos), font, font_scale, font_color, 1, cv2.LINE_AA)
            y_pos += 25

            for joint, angle in angles_data.items():
                angle_range = calculate_angle_range(angle)
                cv2.putText(image, f"{joint}:", (x_col_joint, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(image, str(int(angle)), (x_col_orig, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(image, str(angle_range[0]), (x_col_min, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(image, str(angle_range[1]), (x_col_max, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                y_pos += 30

        cv2.imshow('Workout Analysis', image)
        
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if paused and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angles_data = {
                    "L Shoulder": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                    "R Shoulder": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]),
                    "L Elbow": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_WRIST]),
                    "R Elbow": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]),
                    "L Armpit": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP]),
                    "R Armpit": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]),
                    "L Waist": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE]),
                    "R Waist": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]),
                    "L Knee": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
                    "R Knee": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]),
                }
                print(f"--- Video Paused at Frame {frame_number}. Press 0-9 to categorize, SPACE to resume. ---")
        
        elif paused and angles_data and ord('0') <= key <= ord('9'):
            category = chr(key)
            if category not in all_workout_data:
                all_workout_data[category] = []
            
            all_workout_data[category].append(angles_data.copy())
            print(f"--> Snapshot saved to Category '{category}'. ({len(all_workout_data[category])} total in this category)")

    calculate_and_print_summary(all_workout_data)

    cap.release()
    cv2.destroyAllWindows()

# --- MODIFIED: The script now uses the global VIDEO_FILE_PATH ---
if __name__ == '__main__':
    analyze_workout_video(VIDEO_FILE_PATH)