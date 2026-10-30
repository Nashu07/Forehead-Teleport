
import cv2
from pathlib import Path

BACKGROUND_PATH = Path("background.jpg")


def main():
    # Load the saved empty-room background.
    background = cv2.imread(str(BACKGROUND_PATH))

    if background is None:
        raise FileNotFoundError(
            f"Cannot load background.jpg at: "
            f"{BACKGROUND_PATH.resolve()}"
        )

    # Open the laptop webcam.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    teleport_mode = False

    print("Background Switch Test")
    print("Press B to switch backgrounds.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam.")
                break

            # Mirror the live camera.
            frame = cv2.flip(frame, 1)

            if teleport_mode:
                height, width = frame.shape[:2]
                output = cv2.resize(
                    background, (width, height)
                )
            else:
                output = frame

            cv2.imshow("Forehead Teleport", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):
                teleport_mode = not teleport_mode
                print(
                    "Teleport mode ON"
                    if teleport_mode
                    else "Teleport mode OFF"
                )

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
