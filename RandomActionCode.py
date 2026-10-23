import requests
import random
import time

ROVER_IP = "192.168.252.107"
BASE_URL = f"http://{ROVER_IP}"

ACTIONS = ["FWD","BACK","RIGHT", "LEFT", "STOP"]

def send_cmd(cmd:str):
    url = f"{BASE_URL}/{cmd}"
    try:
        r = requests.get(url, timeout=2.0)
    except requests.exceptions.RequestException as e:
        print("Error talking to Rover", e)

def rand_act():
    return random.choice(ACTIONS)

def duration_time(duration_sec = 10, step_time = 0.8):
    time_end = time.time() + duration_sec
    last_cmd = "STOP"
    send_cmd("STOP")
    while time.time() < time_end:
        action = rand_act()

        send_cmd(action)
        last_cmd = action

        time.sleep(step_time)

    send_cmd("STOP")

if __name__ == "__main__":
    duration_time()
