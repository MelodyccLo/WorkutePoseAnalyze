import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import glob # For listing files
import shutil # For moving files

# ================================================
#           CONFIGURATION PARAMETERS
# ================================================
# --- 1. Folder for incoming videos from the API ---
INCOMING_VIDEO_FOLDER = "incoming_videos_for_analysis"
# --- 2. Folder to move processed videos to ---
PROCESSED_VIDEO_FOLDER = "processed_videos"

# --- 3. Set the desired width of the suggested angle range ---
ANGLE_RANGE_WIDTH = 20

# --- ADD THIS LINE ---
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
# ================================================

# Ensure processed video folder exists
os.makedirs(PROCESSED_VIDEO_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# --- Your existing helper functions ---
def calculate_angle_range(angle, delta=10, round_to=5):
    """
    Calculates a clean, rounded range around an angle.
    """
    raw_lower_bound = 0
    raw_upper_bound = 0

    if (angle - delta) < 0:
        raw_lower_bound = 0
        raw_upper_bound = ANGLE_RANGE_WIDTH
    elif (angle + delta) > 180:
        raw_lower_bound = 180 - ANGLE_RANGE_WIDTH
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
        joint_min_max = {joint: {'min': 180, 'max': 0} for joint in joint_names}

        for data_snapshot in data_list:
            for joint, angle in data_snapshot.items():
                joint_sums[joint] += angle
                joint_min_max[joint]['min'] = min(joint_min_max[joint]['min'], angle)
                joint_min_max[joint]['max'] = max(joint_min_max[joint]['max'], angle)

        joint_averages = {joint: total / num_entries for joint, total in joint_sums.items()}

        print(f"\n--- Summary for Category: {category} ({num_entries} entries) ---")
        print(f"{'Angle:':<15} {'Average':>10} {'Min':>5} {'Max':>5} {'ActualMin':>10} {'ActualMax':>10}")
        print("-" * 75)

        for joint in joint_names:
            avg_angle = joint_averages[joint]
            avg_range = calculate_angle_range(avg_angle)
            print(f"{joint + ':':<15} {int(avg_angle):>10} {avg_range[0]:>5} {avg_range[1]:>5} {int(joint_min_max[joint]['min']):>10} {int(joint_min_max[joint]['max']):>10}")
        print("-" * 75)


def analyze_workout_video_interactive(video_path):
    """
    Analyzes a workout video, logs data, allows categorization, and provides a final summary.
    This is your original interactive analysis function.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print(f"\n--- Analyzing video: {os.path.basename(video_path)} ---")
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

            # Draw landmarks only if detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            else:
                cv2.putText(image, "No Pose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Overlay angle data if paused AND landmarks are detected
        if paused and results.pose_landmarks: # Only show overlay if paused and pose is detected
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
            if paused: # If paused, try to calculate angles if landmarks exist
                if results.pose_landmarks:
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
                else:
                    print(f"--- Video Paused at Frame {frame_number}. No pose detected to categorize. Press SPACE to resume. ---")
                    angles_data = {} # Clear angles if no pose

        elif paused and angles_data and ord('0') <= key <= ord('9'):
            category = chr(key)
            if category not in all_workout_data:
                all_workout_data[category] = []

            all_workout_data[category].append(angles_data.copy())
            print(f"--> Snapshot saved to Category '{category}'. ({len(all_workout_data[category])} total in this category)")

    calculate_and_print_summary(all_workout_data)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(f"Looking for videos in: {INCOMING_VIDEO_FOLDER}")
    # Get a list of video files in the incoming folder
    video_files = glob.glob(os.path.join(INCOMING_VIDEO_FOLDER, '*'))
    video_files = [f for f in video_files if os.path.isfile(f) and f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]

    if not video_files:
        print(f"No video files found in '{INCOMING_VIDEO_FOLDER}'. Please upload a video via the API first.")
    else:
        # Sort files by modification time (newest first)
        video_files.sort(key=os.path.getmtime, reverse=True)
        # Process the newest video first (or you could loop through all)
        video_to_process = video_files[0]

        analyze_workout_video_interactive(video_to_process)

        # After analysis, move the video to the processed folder
        processed_path = os.path.join(PROCESSED_VIDEO_FOLDER, os.path.basename(video_to_process))
        try:
            shutil.move(video_to_process, processed_path)
            print(f"Video moved to processed folder: {processed_path}")
        except Exception as e:
            print(f"Error moving video to processed folder: {e}")

    print("\nWorkout analyzer finished.")