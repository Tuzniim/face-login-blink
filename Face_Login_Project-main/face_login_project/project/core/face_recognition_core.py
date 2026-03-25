# core/face_recognition.py - detect + compare face

import face_recognition
import numpy as np
import cv2
import logging
from config import FACE_RECOGNITION_TOLERANCE, FACE_ENCODING_MODEL

logger = logging.getLogger(__name__)


def detect_faces(frame):
    """
    ตรวจจับตำแหน่งใบหน้าในภาพ
    Returns: list of face locations [(top, right, bottom, left), ...]
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    return face_locations


def encode_face_from_frame(frame):
    """
    แปลงใบหน้าจาก frame กล้อง → encoding vector
    Returns: encoding (numpy array) หรือ None ถ้าไม่เจอหน้า
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        logger.warning("ไม่พบใบหน้าในภาพ")
        return None

    encodings = face_recognition.face_encodings(
        rgb_frame,
        known_face_locations=face_locations,
        model=FACE_ENCODING_MODEL
    )

    if encodings:
        return encodings[0]  # คืนค่า encoding แรก (หน้าแรก)
    return None


def compare_faces(known_encoding, unknown_encoding):
    """
    เปรียบเทียบ encoding สองตัว
    Returns: (is_match: bool, distance: float)
    """
    if known_encoding is None or unknown_encoding is None:
        return False, 1.0

    # แปลง list → numpy array ถ้าจำเป็น
    if isinstance(known_encoding, list):
        known_encoding = np.array(known_encoding)
    if isinstance(unknown_encoding, list):
        unknown_encoding = np.array(unknown_encoding)

    results = face_recognition.compare_faces(
        [known_encoding],
        unknown_encoding,
        tolerance=FACE_RECOGNITION_TOLERANCE
    )
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

    is_match = bool(results[0])
    logger.info(f"Face comparison: match={is_match}, distance={distance:.4f}")
    return is_match, float(distance)


def find_best_match(known_encodings_dict, unknown_encoding):
    """
    หาคนที่ตรงที่สุดจาก dict ของ encodings
    Args:
        known_encodings_dict: {"username": np.array(...), ...}
        unknown_encoding: np.array
    Returns: (username: str | None, distance: float)
    """
    if not known_encodings_dict or unknown_encoding is None:
        return None, 1.0

    best_match = None
    best_distance = 1.0

    for username, encoding in known_encodings_dict.items():
        is_match, distance = compare_faces(encoding, unknown_encoding)
        if is_match and distance < best_distance:
            best_distance = distance
            best_match = username

    return best_match, best_distance


def draw_face_box(frame, face_locations, label="", color=(0, 255, 0)):
    """
    วาดกรอบรอบใบหน้าพร้อม label
    """
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        if label:
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
    return frame
