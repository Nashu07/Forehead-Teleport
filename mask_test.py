
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

MODEL_PATH = Path("models/selfie_segmenter.tflite")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

options = ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH.resolve())
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_confidence_masks=True,
    output_category_mask=True,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

start_time = time.monotonic()
last_timestamp = -1

try:
    with ImageSegmenter.create_from_options(options) as segmenter:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = segmenter.segment_for_video(
                mp_image, timestamp
            )

            if result.confidence_masks:
                mask = result.confidence_masks[0].numpy_view()
                mask = np.squeeze(mask)

                # Show the raw mask without changing its meaning.
                mask = np.clip(mask, 0.0, 1.0)
                mask_display = (mask * 255).astype(np.uint8)

                mask_display = cv2.resize(
                    mask_display,
                    (frame.shape[1], frame.shape[0])
                )

                cv2.imshow("Original Webcam", frame)
                cv2.imshow("AI Person Mask", mask_display)

            elif result.category_mask:
                mask = result.category_mask.numpy_view()
                mask = np.squeeze(mask)

                mask_display = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                cv2.imshow("Original Webcam", frame)
                cv2.imshow("AI Category Mask", mask_display)

            else:
                print("No segmentation mask was returned.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
