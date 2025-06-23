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

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points in a 0-180 degree range.

    Args:
        a: First point.
        b: Second point (vertex).
        c: Third point.

    Returns:
        The angle in degrees.
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

    Args:
        video_path: The path to the workout video file.
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
            cv2.rectangle(overlay, (10, 10), (220, 350), (66, 117, 245), -1)
            alpha = 0.7
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            y_pos = 40
            for joint, angle in angles_data.items():
                cv2.putText(image, f"{joint}: {int(angle)} ", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 30

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
                
                # --- CHANGED: Reordered the dictionary to match your request ---
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
                # --- End of change ---
                
                print(f"--- Angle Data Logged at Frame {frame_number} ---")
                for joint, angle in angles_data.items():
                    print(f"  {joint}: {int(angle)}")
                print("----------------------------------------\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'test2.mov'
    analyze_workout_video(video_path)