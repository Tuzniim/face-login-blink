# core/blink_detection.py - EAR + Blink Detection (Anti-Spoofing)

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import logging
import time

from config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    BLINK_REQUIRED, BLINK_TIMEOUT,
    SHAPE_PREDICTOR_PATH
)

logger = logging.getLogger(__name__)

# dlib detector & predictor (loaded once)
_detector = dlib.get_frontal_face_detector()
_predictor = None


def _load_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
            logger.info("Shape predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shape predictor: {e}")
            raise RuntimeError(
                "shape_predictor_68_face_landmarks.dat not found\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
    return _predictor


def eye_aspect_ratio(eye):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Low value = eye closed
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_landmarks(shape):
    """
    Extract left and right eye landmarks from 68 facial points.
    Returns: (left_eye, right_eye) numpy arrays
    """
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]
    return left_eye, right_eye


class BlinkDetector:
    """
    Blink detection for Anti-Spoofing.
    """

    def __init__(self):
        self.blink_counter = 0       # total blink count
        self.frame_counter = 0       # consecutive frames with eye closed
        self.ear_history = []        # EAR history
        self._predictor = None

    def reset(self):
        """Reset counters"""
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history = []

    def process_frame(self, frame):
        """
        Analyze a frame and update the blink counter.
        Returns: {
            "ear": float,
            "blink_count": int,
            "is_blinking": bool,
            "face_found": bool,
            "annotated_frame": np.ndarray
        }
        """
        predictor = _load_predictor()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector(gray, 0)

        result = {
            "ear": 0.0,
            "blink_count": self.blink_counter,
            "is_blinking": False,
            "face_found": len(faces) > 0,
            "annotated_frame": frame.copy()
        }

        if not faces:
            return result

        # Use the first detected face
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye, right_eye = get_eye_landmarks(shape)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)

        # Draw eye contours
        annotated = result["annotated_frame"]
        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        cv2.drawContours(annotated, [left_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(annotated, [right_hull], -1, (0, 255, 0), 1)

        # Detect blink
        if ear < EAR_THRESHOLD:
            self.frame_counter += 1
            result["is_blinking"] = True
        else:
            if self.frame_counter >= EAR_CONSEC_FRAMES:
                self.blink_counter += 1
                logger.info(f"Blink detected #{self.blink_counter} (EAR: {ear:.3f})")
            self.frame_counter = 0

        result["ear"] = ear
        result["blink_count"] = self.blink_counter

        # Display EAR on frame
        cv2.putText(annotated, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, f"Blinks: {self.blink_counter}/{BLINK_REQUIRED}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return result

    def wait_for_blinks(self, cap, required=None, timeout=None, callback_frame=None):
        """
        Wait until the required number of blinks is detected.
        Args:
            cap: cv2.VideoCapture object
            required: number of blinks required (default from config)
            timeout: timeout in seconds (default from config)
            callback_frame: function for realtime frame callback (show in GUI)
        Returns: (success: bool, frames_annotated: list)
        """
        required = required or BLINK_REQUIRED
        timeout = timeout or BLINK_TIMEOUT
        self.reset()

        start_time = time.time()
        preview_frames = []

        logger.info(f"Waiting for {required} blink(s) (timeout: {timeout}s)")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning("Blink detection timeout")
                return False, preview_frames

            ret, frame = cap.read()
            if not ret:
                continue

            result = self.process_frame(frame)
            preview_frames.append(result["annotated_frame"])

            # Display countdown
            remaining = int(timeout - elapsed)
            cv2.putText(result["annotated_frame"],
                        f"Please blink {required} time(s) ({remaining}s)",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            if callback_frame:
                callback_frame(result["annotated_frame"])

            if result["blink_count"] >= required:
                logger.info(f"Completed {required} blink(s) ✓")
                return True, preview_frames

        return False, preview_frames