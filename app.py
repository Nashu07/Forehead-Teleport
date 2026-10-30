
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pathlib import Path

# ==========================================
# 1. MODEL PATHS
# ==========================================

models = Path("models")

hand_model = models / "hand_landmarker.task"
face_model = models / "face_landmarker.task"
segment_model = models / "selfie_segmenter.tflite"
background_path = Path("background.jpg")

for path in (hand_model, face_model, segment_model):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

# ==========================================
# 2. CONFIGURATION
# ==========================================

TRIGGER_DISTANCE = 0.14
ANIMATION_SPEED = 1.2
MAX_PARTICLES = 1800

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
RunningMode = Vision.RunningMode

hand_options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(hand_model.resolve())
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

face_options = Vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(face_model.resolve())
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

segment_options = Vision.ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path=str(segment_model.resolve())
    ),
    running_mode=RunningMode.VIDEO,
    output_confidence_masks=True
)

# ==========================================
# 3. CAMERA AND BACKGROUND
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

background = None

if background_path.exists():
    background = cv2.imread(str(background_path))

    if background is None:
        raise RuntimeError("Could not load background.jpg")

progress = 0.0

last_time = time.monotonic()
start_time = time.monotonic()
last_timestamp = -1

particles = None
particle_colors = None
particle_velocities = None
particle_sizes = None
dissolve_noise = None

# ==========================================
# 4. PARTICLE GENERATOR
# ==========================================

def create_particles(frame, mask):

    height, width = frame.shape[:2]

    person_pixels = np.argwhere(mask > 0.45)

    if len(person_pixels) == 0:
        return None, None, None, None

    count = min(MAX_PARTICLES, len(person_pixels))

    selected = np.random.choice(
        len(person_pixels),
        count,
        replace=False
    )

    points = person_pixels[selected]

    ys = points[:, 0].astype(np.float32)
    xs = points[:, 1].astype(np.float32)

    colors = frame[
        points[:, 0],
        points[:, 1]
    ].copy()

    center_x = np.mean(xs)
    center_y = np.mean(ys)

    dx = xs - center_x
    dy = ys - center_y

    lengths = np.sqrt(dx * dx + dy * dy) + 0.001

    vx = dx / lengths
    vy = dy / lengths - 0.35

    speeds = np.random.uniform(55, 145, count)

    vx *= speeds
    vy *= speeds

    vx += np.random.uniform(-35, 35, count)
    vy += np.random.uniform(-35, 35, count)

    sizes = np.random.randint(1, 4, count)

    particle_positions = np.column_stack((xs, ys))
    velocities = np.column_stack((vx, vy))

    return (
        particle_positions,
        colors,
        velocities,
        sizes
    )


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

try:

    with (
        Vision.HandLandmarker.create_from_options(
            hand_options
        ) as hands,

        Vision.FaceLandmarker.create_from_options(
            face_options
        ) as faces,

        Vision.ImageSegmenter.create_from_options(
            segment_options
        ) as segmenter
    ):

        print("FOREHEAD TELEPORT - DUST DISSOLVE")
        print("----------------------------------")
        print("1. Leave the camera view.")
        print("2. Press B to capture the empty room.")
        print("3. Return to the camera.")
        print("4. Move your index finger toward your temple.")
        print("5. Your body dissolves into dust particles.")
        print("6. Move your finger away to return.")
        print("Press Q to quit.")

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            now = time.monotonic()
            delta_time = min(now - last_time, 0.05)
            last_time = now

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                int((now - start_time) * 1000),
                last_timestamp + 1
            )

            last_timestamp = timestamp

            # ======================================
            # 6. HAND AND FACE DETECTION
            # ======================================

            hand_result = hands.detect_for_video(
                image, timestamp
            )

            face_result = faces.detect_for_video(
                image, timestamp
            )

            gesture_detected = False

            if (
                face_result.face_landmarks
                and hand_result.hand_landmarks
            ):

                face = face_result.face_landmarks[0]

                left_temple = face[127]
                right_temple = face[356]

                for hand in hand_result.hand_landmarks:

                    tip = hand[8]

                    distance_left = math.hypot(
                        (tip.x - left_temple.x) * width,
                        (tip.y - left_temple.y) * height
                    )

                    distance_right = math.hypot(
                        (tip.x - right_temple.x) * width,
                        (tip.y - right_temple.y) * height
                    )

                    if (
                        distance_left < height * TRIGGER_DISTANCE
                        or distance_right < height * TRIGGER_DISTANCE
                    ):

                        gesture_detected = True
                        break

            # ======================================
            # 7. CAPTURE EMPTY ROOM
            # ======================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):

                if gesture_detected:

                    print("Move your finger away first.")

                else:

                    background = frame.copy()

                    if cv2.imwrite(
                        str(background_path),
                        background
                    ):

                        print("Empty-room background saved!")

                    else:

                        print("Could not save background.")

            # ======================================
            # 8. PERSON SEGMENTATION
            # ======================================

            raw_mask = None

            if background is not None:

                segment_result = segmenter.segment_for_video(
                    image,
                    timestamp
                )

                if segment_result.confidence_masks:

                    raw_mask = np.squeeze(
                        segment_result.confidence_masks[0].numpy_view()
                    ).astype(np.float32)

                    raw_mask = cv2.resize(
                        raw_mask,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR
                    )

                    raw_mask = np.clip(
                        raw_mask,
                        0.0,
                        1.0
                    )

            # ======================================
            # 9. ANIMATION PROGRESS
            # ======================================

            previous_progress = progress

            if gesture_detected and background is not None:

                progress = min(
                    1.0,
                    progress + delta_time * ANIMATION_SPEED
                )

            else:

                progress = max(
                    0.0,
                    progress - delta_time * ANIMATION_SPEED
                )

            # Create particles at the beginning.
            if (
                previous_progress == 0.0
                and progress > 0.0
                and raw_mask is not None
            ):

                (
                    particles,
                    particle_colors,
                    particle_velocities,
                    particle_sizes
                ) = create_particles(frame, raw_mask)

                dissolve_noise = np.random.random(
                    (height, width)
                ).astype(np.float32)

            # ======================================
            # 10. DUST DISSOLVE EFFECT
            # ======================================

            if background is not None and progress > 0:

                bg = cv2.resize(
                    background,
                    (width, height)
                )

                if (
                    raw_mask is not None
                    and dissolve_noise is not None
                ):

                    # Gradually dissolve the person.
                    keep_mask = (
                        dissolve_noise > progress
                    ).astype(np.float32)

                    alpha = (
                        raw_mask * keep_mask
                    )[:, :, None]

                    output = (
                        frame.astype(np.float32) * alpha
                        + bg.astype(np.float32) * (1.0 - alpha)
                    )

                    output = np.clip(
                        output,
                        0,
                        255
                    ).astype(np.uint8)

                else:

                    output = bg.copy()

                # ==================================
                # 11. PARTICLE FADE FIX
                # ==================================

                if (
                    particles is not None
                    and particle_colors is not None
                    and particle_velocities is not None
                ):

                    # Particles disappear completely
                    # before the dissolve reaches 100%.
                    fade = max(
                        0.0,
                        1.0 - progress / 0.85
                    )

                    # Never draw invisible particles.
                    if fade > 0.02:

                        for i in range(len(particles)):

                            x = int(
                                particles[i, 0]
                                + particle_velocities[i, 0] * progress
                            )

                            y = int(
                                particles[i, 1]
                                + particle_velocities[i, 1] * progress
                            )

                            if (
                                0 <= x < width
                                and 0 <= y < height
                            ):

                                color = (
                                    particle_colors[i].astype(np.float32)
                                    * fade
                                ).astype(np.uint8)

                                cv2.circle(
                                    output,
                                    (x, y),
                                    int(particle_sizes[i]),
                                    tuple(int(c) for c in color),
                                    -1
                                )

                message = "DUST TELEPORT"

            else:

                output = frame.copy()

                if background is None:
                    message = "Press B to capture empty room"
                else:
                    message = "Move finger toward temple"

            # ======================================
            # 12. DISPLAY
            # ======================================

            

            cv2.imshow(
                "Forehead Teleport - Dust Effect",
                output
            )

            if key == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()
