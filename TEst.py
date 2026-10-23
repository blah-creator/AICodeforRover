import cv2
import time

CAMERA = "http://192.168.254.57:81/stream"

def main():
    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {CAMERA}")

    last_t = time.time()
    frame_count = 0

    while True:
        # flush a few frames
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        now = time.time()
        dt = now - last_t
        last_t = now
        frame_count += 1

        cv2.putText(
            frame,
            f"dt={dt:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Raw Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()