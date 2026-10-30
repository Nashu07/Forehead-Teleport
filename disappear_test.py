
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

MODEL_PATH = Path("models/selfie_segmenter.tflite")
BACKGROUND_PATH = Path("background.jpg")


def main():
    # Check that the required files exist.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"Background not found: {BACKGROUND_PATH.resolve()}"
        )

    # Load the empty-room background.
    background = cv2.imread(str(BACKGROUND_PATH))

    if background is None:
        raise RuntimeError("Could not load background.jpg")

    # Configure MediaPipe image segmentation.
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_confidence_masks=True,
    )

    # Open the laptop webcam.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    start_time = time.monotonic()
    last_timestamp = -1

    try:
        with mp.tasks.vision.ImageSegmenter.create_from_options(
            options
        ) as segmenter:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam.")
                    break

                # Mirror the webcam view.
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                # Resize the saved background to match the webcam.
                bg = cv2.resize(
                    background,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Prepare the frame for MediaPipe.
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                timestamp = max(
                    int((time.monotonic() - start_time) * 1000),
                    last_timestamp + 1,
                )
                last_timestamp = timestamp

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = segmenter.segment_for_video(
                    mp_image,
                    timestamp,
                )

                if not result.confidence_masks:
                    cv2.imshow("Original Webcam", frame)
                    cv2.imshow("Disappearance Test", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    continue

                # Read the AI segmentation mask.
                # High values indicate the person.
                raw_mask = np.squeeze(
                    result.confidence_masks[0].numpy_view()
                ).astype(np.float32)

                mask = cv2.resize(
                    raw_mask,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

                mask = np.clip(mask, 0.0, 1.0)

                # White/person = 1; black/background = 0.
                person_mask = (
                    mask > 0.5
                ).astype(np.float32)

                # Expand the mask slightly to cover edge shadows.
                person_mask_uint8 = (
                    person_mask * 255
                ).astype(np.uint8)

                kernel = np.ones((7, 7), np.uint8)

                person_mask_uint8 = cv2.dilate(
                    person_mask_uint8,
                    kernel,
                    iterations=1,
                )

                person_mask = (
                    person_mask_uint8.astype(np.float32) / 255.0
                )

                # Slightly soften the boundary.
                person_mask = cv2.GaussianBlur(
                    person_mask,
                    (5, 5),
                    0,
                 )

                # Make the mask compatible with 3-channel images.
                person_mask = person_mask[:, :, None]

                frame_float = frame.astype(np.float32)
                bg_float = bg.astype(np.float32)

                # IMPORTANT:
                # Replace the detected person with the saved background.
                output = (
                    frame_float * (1.0 - person_mask)
                    + bg_float * person_mask
                )

                output = np.clip(
                    output,
                    0,
                    255,
                ).astype(np.uint8)

                # Display the mask and final result.
                mask_view = (
                    mask * 255
                ).astype(np.uint8)

                cv2.imshow("Original Webcam", frame)
                cv2.imshow("AI Person Mask", mask_view)
                cv2.imshow("Disappearance Test", output)

                # Press Q to close all windows.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
