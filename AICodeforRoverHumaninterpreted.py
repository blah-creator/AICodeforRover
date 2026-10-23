import requests
import time
from pathlib import Path
import csv

ROVER_IP = "192.168.254.137"
BASE_URL = f"http://{ROVER_IP}"
CSV_PATH = Path("rover_data.csv")
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
        r = requests.get(url, timeout=0.2)
    except requests.exceptions.RequestException as e:
        print("Error talking to Rover", e)

def ensure_csv_header(path: Path):
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "label"])

def main():
    ensure_csv_header(CSV_PATH)
    print("Controls:")
    print("  w = FWD, s = BACK, a = LEFT, d = RIGHT, x = STOP, q = quit")
    print("After each move you'll label it as good (g) or bad (b).")
    print()

    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)

        while True:
            key = input("Choose your direction w,s,a,d,x, or q: ").strip().lower()
            if key == "q":
                send_cmd("STOP")
                print("Quitting")
                break
            if key not in ACTIONS:
                print("Invalid key inputted")
                continue
            action = ACTIONS[key]
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
            writer.writerow([timestamp, action, label_text])
            f.flush()
            print(f"Logged action was {action}. Logged label was {label_text}")
            print()


if __name__ == "__main__":
 main()
