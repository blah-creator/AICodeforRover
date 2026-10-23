import requests
import time
from pathlib import Path
import csv
import cv2
import re

ROVER_IP = "192.168.254.137"
BASE_URL = f"http://{ROVER_IP}"

CAMERA = "http://192.168.254.57:81/stream"
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "rover_data_camera.csv"
ACTIONS = {
    "w": "FWD",
    "s": "BACK",
    "d": "RIGHT",
    "a": "LEFT",
    "x": "STOP"
}

def send_cmd(cmd:str):
    url = f"{BASE_URL}/{cmd}"
    try:
        r = requests.get(url, timeout=1)
    except requests.exceptions.RequestException as e:
        print("Error talking to Rover", e)

def ensure_csv_header(path: Path):
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    IMAGE_DIR.mkdir(parents = True, exist_ok = True)

    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "label", "image_path"])

def open_camera():
    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open Camera Source: {CAMERA}")
    return cap

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

def get_next_index():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    highest = 0
    pattern = re.compile(r"test_(\d+)\.jpg")

    for f in IMAGE_DIR.glob("test_*.jpg"):
        m = pattern.match(f.name)
        if m:
            num = int(m.group(1))
            highest = max(highest, num)

    return highest

def main():
    ensure_csv_header(CSV_PATH)
    cap = open_camera()

    print("Controls:")
    print("  w = FWD, s = BACK, a = LEFT, d = RIGHT, x = STOP, q = quit")
    print("After each move you'll label it as good (g) or bad (b).")
    print()

    test_idx = get_next_index()

    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        try:
            while True:
                for i in range(15):
                    cap, _ = get_frame(cap)

                cap, frame = get_frame(cap)

                cv2.imshow("Rover Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit from Camera Window")
                    break

                key = input("Choose your direction w,s,a,d,x, or q: ").strip().lower()

                if key == "q":
                    send_cmd("STOP")
                    print("Quitting")
                    break

                if key not in ACTIONS:
                    print("Invalid key inputted")
                    continue

                action = ACTIONS[key]

                #time.sleep(2)

                test_idx += 1
                img_name = f"test_{test_idx:03d}.jpg"
                img_path = IMAGE_DIR / img_name
                cv2.imwrite(str(img_path), frame)
                print(f"Saved Image is located at {img_path}")

                send_cmd(action)
                time.sleep(0.5)
                send_cmd("STOP")

                while True:
                    label = input("Was that move good(g) or bad(b): ").strip().lower()
                    if label in ("g", "b"):
                        label_text = "good" if label == "g" else "bad"
                        break
                    print("Please enter g or b")
                timestamp = time.time()
                writer.writerow([timestamp, action, label_text, str(img_path)])
                f.flush()
                print(f"Logged action was {action}. Logged label was {label_text}. Image is located at {img_path}")
                print()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            send_cmd("STOP")
            print("All Done")

if __name__ == "__main__":
    main()
