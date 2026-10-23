CAM_IP = "http://192.168.254.57:81/stream"
ROV_IP = "192.168.254.137"

import requests
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import csv
from collections import deque, Counter
from ultralytics import YOLO

VOTE_WINDOW = 7          # number of recent predictions to vote over
MIN_CONF_FOR_MOVE = 0.35 # raise slightly from 0.30 if you want safer
TURN_BIAS_MARGIN = 0.08  # if turn prob is within this of FWD, choose turn
STEP_TIME = 0.35         # shorter steps = more responsive


ROVER_IP = ROV_IP
BASE_URL = f"http://{ROVER_IP}"

CAMERA = CAM_IP
DATA_DIR = Path("data")

MODEL_PATH = Path("v2_ai_model.pkl")
LIVE_CSV = DATA_DIR / "live_feedback_rover.csv"
LIVE_IMG_DIR = DATA_DIR / "live_images"

IMAGE_SIZE = 16

ACTIONS = {"FWD": 0,"BACK": 1,"RIGHT": 2,"LEFT": 3}
IDX_TO_ACTION = {v: k for k, v in ACTIONS.items()}

CONFIDENCE_THRESHOLD = 0.3

def send_cmd(cmd:str):
    url = f"{BASE_URL}/{cmd}"
    try:
        r = requests.get(url, timeout=1)
    except requests.exceptions.RequestException as e:
        print("Error talking to Rover", e)

def open_camera():
    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open Camera Source: {CAMERA}")
    return cap

def pre_process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #CONVERTS THE IMAGE TO GRAY AND IT DEPENDS ON BRIGHTNESS FOR TEXTURE
    small = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE)) #RESIZES IT TO THE SET IMAGE_SIZE
    arr = small.astype("float32").flatten() / 255.0 #MAKES IT A 32-BIT NUMBER THAT THE AI READS AND PUTS IT INTO A SCALE OF 0-1
    return arr.reshape(1,-1) #TURNS IT FROM 256 VALUES INTO [ [256 VALUES] ]

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream dropped — reconnecting camera...")
        cap.release()
        time.sleep(0.5)
        cap = cv2.VideoCapture(CAMERA, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError("Camera could not reconnect")

        print("✅ Camera reconnected")
        ret, frame = cap.read()

    return cap, frame

def main():
    global fixed_action
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    live_file = LIVE_CSV.open("a", newline="")
    writer = csv.writer(live_file)

    if LIVE_CSV.stat().st_size == 0:
        writer.writerow(["timestamp", "action", "label", "image_path"])

    if not MODEL_PATH.exists():
        raise FileNotFoundError("AI Model Does Not Exist")
    model = joblib.load(MODEL_PATH)
    yolo = YOLO("yolov8n-seg.pt")
    print("YOLO generated")
    print(f"Loaded model: {MODEL_PATH}")

    has_proba = hasattr(model, "predict_proba") #CHECKS IF THE MODEL HAS THE PREDICTION MODEL AND RETURNS A BOOLEAN(TRUE OR FALSE)
    print(f"Model Has a Prediction Model: {has_proba}")

    cap = open_camera()
    print("Press 'x' to STOP the rover and 'q' to quit.")

    last_action = "STOP"

    try:
        while True:
            for _ in range(50):
                cap.read()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab image from Camera")
                cap, frame = get_frame(cap)
                if frame is None:
                    continue

            # --- YOLO OBJECT DETECTION ---
            results = yolo.predict(frame, verbose=False)[0]

            detected_classes = []
            for box in results.boxes:
                cls_id = int(box.cls)
                cls_name = yolo.names[cls_id]
                conf = float(box.conf)

                if conf < 0.40:
                    continue

                detected_classes.append(cls_name)

            # --- DECISION LOGIC ---
            if "person" in detected_classes:
                action_now = "STOP"
            elif "chair" in detected_classes:
                action_now = "LEFT"
            elif "bottle" in detected_classes:
                action_now = "RIGHT"
            else:
                action_now = "FWD"

            # --- SEND COMMAND ---
            send_cmd(action_now)
            last_action = action_now
            time.sleep(STEP_TIME)
            send_cmd("STOP")

            # --- DISPLAY YOLO FRAME ---
            annotated = results.plot()
            cv2.imshow("Rover Camera", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                send_cmd("STOP")
                print("Quitting")
                break
            if key == ord('x'):
                send_cmd("STOP")
                last_action = "STOP"
                print("STOPPING")

            time.sleep(0.1)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        send_cmd("STOP")
        print("All Done")

if __name__ == "__main__":
    main()