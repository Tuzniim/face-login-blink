# database/db_connect.py - เชื่อมต่อ Supabase

import logging
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

_supabase_client: Client = None


def get_client() -> Client:
    """
    คืนค่า Supabase client (Singleton)
    """
    global _supabase_client
    if _supabase_client is None:
        try:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("เชื่อมต่อ Supabase สำเร็จ")
        except Exception as e:
            logger.error(f"เชื่อมต่อ Supabase ล้มเหลว: {e}")
            raise
    return _supabase_client


def test_connection():
    """
    ทดสอบการเชื่อมต่อ Supabase
    Returns: bool
    """
    try:
        client = get_client()
        # ลองดึงข้อมูลเพื่อทดสอบ
        client.table("users").select("id").limit(1).execute()
        logger.info("ทดสอบการเชื่อมต่อสำเร็จ")
        return True
    except Exception as e:
        logger.error(f"ทดสอบการเชื่อมต่อล้มเหลว: {e}")
        return False
