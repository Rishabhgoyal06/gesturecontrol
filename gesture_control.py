import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model #type: ignore
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


model = load_model("gesture_model.h5")
labels = np.load("labels.npy", allow_pickle=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_id = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, frame_id)
    frame_id += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmark_list = []
        for lm in hand:
            landmark_list.extend([lm.x, lm.y, lm.z])

        data = np.array(landmark_list).reshape(1, -1)

        pred = model.predict(data, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        gesture = labels[class_id]

        cv2.putText(frame, f"{gesture} ({confidence*100:.1f}%)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        print("Gesture:", gesture)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

