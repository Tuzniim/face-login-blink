# services/train_service.py - encode รูปทั้งหมดและบันทึกลง DB

import logging
import os
from pathlib import Path

from core.encoding import encode_user_images
from database.user_model import get_all_users, update_face_encoding, user_exists

logger = logging.getLogger(__name__)


def train_single_user(username: str) -> dict:
    """
    Encode ใบหน้าของ user คนเดียวและบันทึกลง DB
    Returns: {"success": bool, "message": str}
    """
    if not user_exists(username):
        return {"success": False, "message": f"ไม่พบ user '{username}' ใน DB"}

    encoding = encode_user_images(username)
    if encoding is None:
        return {
            "success": False,
            "message": f"ไม่สามารถ encode ใบหน้าของ '{username}' ได้"
        }

    success = update_face_encoding(username, encoding)
    if success:
        logger.info(f"Train '{username}' สำเร็จ")
        return {"success": True, "message": f"อัปเดต encoding ของ '{username}' สำเร็จ"}
    else:
        return {"success": False, "message": f"บันทึก encoding ล้มเหลว"}


def train_all_users() -> dict:
    """
    Encode ใบหน้าของทุก user ใน DB และบันทึก
    Returns: {
        "total": int,
        "success": int,
        "failed": list
    }
    """
    users = get_all_users()
    results = {"total": len(users), "success": 0, "failed": []}

    for user in users:
        username = user["username"]
        result = train_single_user(username)
        if result["success"]:
            results["success"] += 1
        else:
            results["failed"].append(username)
            logger.warning(f"Train '{username}' ล้มเหลว: {result['message']}")

    logger.info(
        f"Train เสร็จสิ้น: {results['success']}/{results['total']} สำเร็จ"
    )
    return results


def preload_encodings() -> dict:
    """
    โหลด encoding ทั้งหมดจาก DB เข้า memory (สำหรับ cache)
    Returns: {"username": np.array, ...}
    """
    from database.user_model import get_all_encodings
    encodings = get_all_encodings()
    logger.info(f"โหลด {len(encodings)} encodings เข้า memory")
    return encodings
