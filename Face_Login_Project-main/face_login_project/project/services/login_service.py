# services/login_service.py - login + blink detection + face comparison

import cv2
import logging
import time
from dataclasses import dataclass

from core.face_recognition_core import (
    encode_face_from_frame,
    find_best_match,
    detect_faces
)
from core.blink_detection import BlinkDetector
from database.user_model import get_all_encodings
from database.log_model import insert_log
from config import (
    BLINK_REQUIRED,
    BLINK_TIMEOUT,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)

logger = logging.getLogger(__name__)


@dataclass
class LoginResult:
    success: bool
    username: str | None
    full_name: str | None
    confidence: float
    blink_count: int
    status: str  # 'success' | 'failed' | 'spoof_detected' | 'timeout' | 'no_face'
    message: str
    elapsed_time: float = 0.0


def login_with_face(callback_frame=None, callback_status=None) -> LoginResult:
    """
    Full login pipeline:
    1. Load known face encodings from database
    2. Open camera and detect face
    3. Perform blink detection (anti-spoofing)
    4. Compare face with database
    5. Save login log
    """
    start_time = time.time()

    def update_status(msg):
        logger.info(msg)
        if callback_status:
            callback_status(msg)

    # Step 1: Load known encodings
    update_status("Loading face data...")
    known_encodings = get_all_encodings()

    if not known_encodings:
        insert_log(None, "failed", message="No registered users")
        return LoginResult(
            success=False,
            username=None,
            full_name=None,
            confidence=0.0,
            blink_count=0,
            status="failed",
            message="No face data found. Please register first."
        )

    # Step 2: Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        return LoginResult(
            success=False,
            username=None,
            full_name=None,
            confidence=0.0,
            blink_count=0,
            status="failed",
            message="Unable to access camera"
        )

    blink_detector = BlinkDetector()
    update_status(f"Please blink {BLINK_REQUIRED} times to verify identity...")

    # Step 3: Blink Detection
    blink_success, _ = blink_detector.wait_for_blinks(
        cap,
        required=BLINK_REQUIRED,
        timeout=BLINK_TIMEOUT,
        callback_frame=callback_frame
    )

    if not blink_success:
        cap.release()
        insert_log(
            None,
            "spoof_detected",
            blink_count=blink_detector.blink_counter,
            message="Blink verification failed / timeout"
        )
        return LoginResult(
            success=False,
            username=None,
            full_name=None,
            confidence=0.0,
            blink_count=blink_detector.blink_counter,
            status="spoof_detected",
            message="Anti-spoofing failed. Please blink properly."
        )

    blink_count = blink_detector.blink_counter
    update_status("Anti-spoofing passed ✓ Recognizing face...")

    # Step 4: Capture frame and encode face
    face_encoding = None
    face_timeout = time.time() + 5

    while time.time() < face_timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        if callback_frame:
            callback_frame(frame)

        face_locations = detect_faces(frame)
        if face_locations:
            face_encoding = encode_face_from_frame(frame)
            if face_encoding is not None:
                break

    cap.release()

    if face_encoding is None:
        insert_log(None, "no_face", blink_count=blink_count, message="No face detected")
        return LoginResult(
            success=False,
            username=None,
            full_name=None,
            confidence=0.0,
            blink_count=blink_count,
            status="no_face",
            message="No face detected. Please try again."
        )

    # Step 5: Compare face
    update_status("Matching face...")
    matched_username, distance = find_best_match(known_encodings, face_encoding)
    confidence = round(1.0 - distance, 4)
    elapsed = round(time.time() - start_time, 2)

    if matched_username:
        from database.user_model import get_user_by_username

        user = get_user_by_username(matched_username)
        full_name = user.get("full_name", matched_username) if user else matched_username

        insert_log(
            matched_username,
            "success",
            confidence=confidence,
            blink_count=blink_count,
            message=f"Login success (distance={distance:.4f})"
        )

        return LoginResult(
            success=True,
            username=matched_username,
            full_name=full_name,
            confidence=confidence,
            blink_count=blink_count,
            status="success",
            message=f"Welcome {full_name}!",
            elapsed_time=elapsed
        )
    else:
        insert_log(
            None,
            "failed",
            confidence=confidence,
            blink_count=blink_count,
            message=f"Unknown face (distance={distance:.4f})"
        )

        return LoginResult(
            success=False,
            username=None,
            full_name=None,
            confidence=confidence,
            blink_count=blink_count,
            status="failed",
            message="Face not recognized. Please try again.",
            elapsed_time=elapsed
        )