# core/encoding.py - แปลง face → encoding vector + จัดการไฟล์

import face_recognition
import numpy as np
import os
import json
import logging
from pathlib import Path

from config import DATASET_PATH, FACE_ENCODING_MODEL

logger = logging.getLogger(__name__)


def encode_image_file(image_path):
    """
    โหลดรูปจากไฟล์และแปลงเป็น encoding
    Returns: numpy array หรือ None
    """
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model=FACE_ENCODING_MODEL)
        if encodings:
            logger.info(f"Encode รูป {image_path} สำเร็จ")
            return encodings[0]
        else:
            logger.warning(f"ไม่พบใบหน้าในรูป: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error encoding {image_path}: {e}")
        return None


def encoding_to_list(encoding):
    """
    แปลง numpy array → Python list (สำหรับบันทึก JSON / Supabase)
    """
    if encoding is None:
        return None
    return encoding.tolist()


def list_to_encoding(encoding_list):
    """
    แปลง Python list → numpy array (สำหรับโหลดจาก DB)
    """
    if encoding_list is None:
        return None
    return np.array(encoding_list, dtype=np.float64)


def encode_user_images(username):
    """
    Encode รูปทุกรูปในโฟลเดอร์ dataset/<username>/
    Returns: encoding เฉลี่ย หรือ None
    """
    user_dir = Path(DATASET_PATH) / username
    if not user_dir.exists():
        logger.error(f"ไม่พบโฟลเดอร์: {user_dir}")
        return None

    encodings = []
    supported_ext = [".jpg", ".jpeg", ".png", ".bmp"]

    for img_file in user_dir.iterdir():
        if img_file.suffix.lower() in supported_ext:
            enc = encode_image_file(str(img_file))
            if enc is not None:
                encodings.append(enc)

    if not encodings:
        logger.warning(f"ไม่มี encoding สำหรับ {username}")
        return None

    # คืนค่า encoding เฉลี่ย (ถ้ามีหลายรูป)
    mean_encoding = np.mean(encodings, axis=0)
    logger.info(f"Encode {username} ได้ {len(encodings)} รูป")
    return mean_encoding


def save_encoding_local(username, encoding, save_dir="encodings"):
    """
    บันทึก encoding ลงไฟล์ JSON ในเครื่อง (สำรอง)
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{username}.json")
    data = {
        "username": username,
        "encoding": encoding_to_list(encoding)
    }
    with open(filepath, "w") as f:
        json.dump(data, f)
    logger.info(f"บันทึก encoding ของ {username} ที่ {filepath}")


def load_encoding_local(username, save_dir="encodings"):
    """
    โหลด encoding จากไฟล์ JSON ในเครื่อง
    Returns: numpy array หรือ None
    """
    filepath = os.path.join(save_dir, f"{username}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return list_to_encoding(data["encoding"])
    except Exception as e:
        logger.error(f"โหลด encoding ล้มเหลว: {e}")
        return None


def load_all_encodings_local(save_dir="encodings"):
    """
    โหลด encoding ทั้งหมดจากโฟลเดอร์ local
    Returns: {"username": np.array, ...}
    """
    result = {}
    if not os.path.exists(save_dir):
        return result
    for f in os.listdir(save_dir):
        if f.endswith(".json"):
            username = f[:-5]
            enc = load_encoding_local(username, save_dir)
            if enc is not None:
                result[username] = enc
    return result
