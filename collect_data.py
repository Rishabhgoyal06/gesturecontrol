import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------- FIXED MODEL PATH (IMPORTANT) -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("\n❌ ERROR: hand_landmarker.task file not found!")
    print("Put the file in same folder as this script:")
    print(BASE_DIR)
    exit()

# ----------- Gesture name -----------
GESTURE_NAME = input("Enter gesture name: ")

SAVE_DIR = os.path.join(BASE_DIR, "gesture_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

file_path = os.path.join(SAVE_DIR, f"{GESTURE_NAME}.npy")

# ----------- MediaPipe Setup -----------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ----------- Webcam -----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW prevents black camera on Windows

data = []
frame_id = 0

print("\nPress 'S' to capture sample")
print("Press 'ESC' to finish\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect hand
    result = detector.detect_for_video(mp_image, frame_id)
    frame_id += 1

    # if hand detected
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmark_list = []
        for lm in hand:
            landmark_list.extend([lm.x, lm.y, lm.z])

        # draw points
        h, w, _ = frame.shape
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            data.append(landmark_list)
            print("Captured:", len(data))

    # display info
    cv2.putText(frame,
                f"Gesture: {GESTURE_NAME}  Samples: {len(data)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Gesture Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ----------- Save dataset -----------
np.save(file_path, np.array(data))
print(f"\n✅ Saved {len(data)} samples to:")
print(file_path)
