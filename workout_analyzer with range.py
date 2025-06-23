import cv2
import mediapipe as mp
import numpy as np
import math
import time 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

def calculate_angle_range(angle, delta=10, total_range_width=20, round_to=5):
    """
    Calculates a clean, rounded range around an angle.
    """
    raw_lower_bound = 0
    raw_upper_bound = 0

    if (angle - delta) < 0:
        raw_lower_bound = 0
        raw_upper_bound = total_range_width
    elif (angle + delta) > 180:
        raw_lower_bound = 180 - total_range_width
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

def analyze_workout_video(video_path):
    """
    Analyzes a workout video, logs data to the terminal, and displays it in a specific order.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    paused = False
    angles_data = {}

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
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )
        
        if paused and results.pose_landmarks:
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (320, 380), (66, 117, 245), -1)
            alpha = 0.7
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # --- CODE MODIFIED: Draw each column of text separately for perfect alignment ---
            
            # Define font, scale, color, and separator
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)
            thickness = 2
            separator = "-" * 19

            # Define the starting X-coordinate for each column
            x_col_joint = 20
            x_col_orig = 170
            x_col_min = 220
            x_col_max = 270
            
            # Start position for the first line
            y_pos = 40

            # Draw the header for each column at its specific X-coordinate
            cv2.putText(image, "Angle:", (x_col_joint, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Orig", (x_col_orig, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Min", (x_col_min, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, "Max", (x_col_max, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
            y_pos += 25

            # Draw the separator line
            cv2.putText(image, separator, (20, y_pos), font, font_scale, font_color, 1, cv2.LINE_AA)
            y_pos += 25

            # Draw the data for each joint, column by column
            for joint, angle in angles_data.items():
                angle_range = calculate_angle_range(angle)
                
                # Column 1: Joint Name
                cv2.putText(image, f"{joint}:", (x_col_joint, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                # Column 2: Original Angle
                cv2.putText(image, str(int(angle)), (x_col_orig, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                # Column 3: Min Range
                cv2.putText(image, str(angle_range[0]), (x_col_min, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                # Column 4: Max Range
                cv2.putText(image, str(angle_range[1]), (x_col_max, y_pos), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                y_pos += 30 # Move to the next line

        cv2.imshow('Workout Analysis', image)
        
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            
            if paused and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                left_elbow = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
                right_elbow = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                left_knee = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
                right_knee = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
                left_shoulder = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                right_shoulder = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
                left_armpit = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                right_armpit = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                left_waist = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
                right_waist = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
                
                angles_data = {
                    "L Shoulder": left_shoulder,
                    "R Shoulder": right_shoulder,
                    "L Elbow": left_elbow,
                    "R Elbow": right_elbow,
                    "L Armpit": left_armpit,
                    "R Armpit": right_armpit,
                    "L Waist": left_waist,
                    "R Waist": right_waist,
                    "L Knee": left_knee,
                    "R Knee": right_knee,
                }
                
                # Terminal output remains the same, as it was already perfect
                print(f"--- Angle Data Logged at Frame {frame_number} ---")
                print(f"{'Angle:':<15} {'Original':>10} {'Min':>5} {'Max':>5}")
                print("-" * 38)
                
                for joint, angle in angles_data.items():
                    angle_range = calculate_angle_range(angle)
                    print(f"{joint + ':':<15} {int(angle):>10} {angle_range[0]:>5} {angle_range[1]:>5}")
                print("--------------------------------------\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'test1.mov' 
    analyze_workout_video(video_path)