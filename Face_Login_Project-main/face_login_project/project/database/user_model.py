# database/user_model.py - insert / select users

import logging
import json
from datetime import datetime
from database.db_connect import get_client
from core.encoding import encoding_to_list, list_to_encoding
import numpy as np

logger = logging.getLogger(__name__)

# ---- Supabase Table Schema (สร้างใน Supabase) ----
# CREATE TABLE users (
#   id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#   username TEXT UNIQUE NOT NULL,
#   full_name TEXT,
#   face_encoding JSONB,          -- เก็บ encoding เป็น JSON array
#   created_at TIMESTAMPTZ DEFAULT NOW(),
#   is_active BOOLEAN DEFAULT TRUE
# );

TABLE_NAME = "users"


def create_user(username: str, full_name: str, face_encoding: np.ndarray) -> dict | None:
    """
    เพิ่ม user ใหม่พร้อม face encoding
    Returns: user record หรือ None
    """
    try:
        client = get_client()
        data = {
            "username": username,
            "full_name": full_name,
            "face_encoding": encoding_to_list(face_encoding),
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
        response = client.table(TABLE_NAME).insert(data).execute()
        if response.data:
            logger.info(f"สร้าง user '{username}' สำเร็จ")
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"สร้าง user ล้มเหลว: {e}")
        return None


def get_user_by_username(username: str) -> dict | None:
    """
    ดึงข้อมูล user ตาม username
    Returns: user record หรือ None
    """
    try:
        client = get_client()
        response = (
            client.table(TABLE_NAME)
            .select("*")
            .eq("username", username)
            .eq("is_active", True)
            .single()
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"ดึงข้อมูล user ล้มเหลว: {e}")
        return None


def get_all_users() -> list:
    """
    ดึง user ที่ active ทั้งหมด
    Returns: list of user records
    """
    try:
        client = get_client()
        response = (
            client.table(TABLE_NAME)
            .select("id, username, full_name, face_encoding, created_at")
            .eq("is_active", True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"ดึง users ทั้งหมดล้มเหลว: {e}")
        return []


def get_all_encodings() -> dict:
    """
    ดึง encoding ของ users ทั้งหมด
    Returns: {"username": np.array, ...}
    """
    users = get_all_users()
    encodings = {}
    for user in users:
        if user.get("face_encoding"):
            enc = list_to_encoding(user["face_encoding"])
            if enc is not None:
                encodings[user["username"]] = enc
    return encodings


def user_exists(username: str) -> bool:
    """
    ตรวจสอบว่า username มีอยู่แล้วหรือไม่
    """
    user = get_user_by_username(username)
    return user is not None


def update_face_encoding(username: str, new_encoding: np.ndarray) -> bool:
    """
    อัปเดต face encoding ของ user
    """
    try:
        client = get_client()
        response = (
            client.table(TABLE_NAME)
            .update({"face_encoding": encoding_to_list(new_encoding)})
            .eq("username", username)
            .execute()
        )
        logger.info(f"อัปเดต encoding ของ '{username}' สำเร็จ")
        return bool(response.data)
    except Exception as e:
        logger.error(f"อัปเดต encoding ล้มเหลว: {e}")
        return False


def deactivate_user(username: str) -> bool:
    """
    ปิดการใช้งาน user (soft delete)
    """
    try:
        client = get_client()
        response = (
            client.table(TABLE_NAME)
            .update({"is_active": False})
            .eq("username", username)
            .execute()
        )
        logger.info(f"ปิดการใช้งาน user '{username}' สำเร็จ")
        return bool(response.data)
    except Exception as e:
        logger.error(f"ปิดการใช้งาน user ล้มเหลว: {e}")
        return False
