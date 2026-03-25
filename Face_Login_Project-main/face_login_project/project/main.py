#!/usr/bin/env python3
# main.py - Entry point ของโปรแกรม

import sys
import os
import logging

# เพิ่ม project root ใน Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL

# ─── Setup Logging ───
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def check_requirements():
    """ตรวจสอบ dependencies และ model files"""
    import importlib
    required_modules = [
        "face_recognition", "cv2", "dlib",
        "numpy", "scipy", "supabase", "imutils"
    ]
    missing = []
    for mod in required_modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        logger.error(f"ขาด module: {', '.join(missing)}")
        print(f"❌ กรุณาติดตั้ง: pip install {' '.join(missing)}")
        sys.exit(1)

    # ตรวจสอบ dlib model
    from config import SHAPE_PREDICTOR_PATH
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"""
⚠️  ไม่พบไฟล์ {SHAPE_PREDICTOR_PATH}

กรุณาดาวน์โหลด:
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  mkdir -p models && mv shape_predictor_68_face_landmarks.dat models/
        """)
        sys.exit(1)

    logger.info("✅ ตรวจสอบ requirements ผ่าน")


def main():
    print("=" * 50)
    print("🔐 Face Recognition Login System")
    print("   with Blink Detection Anti-Spoofing")
    print("=" * 50)

    # ตรวจสอบ requirements
    check_requirements()

    # ทดสอบ Supabase connection
    logger.info("ทดสอบการเชื่อมต่อ Supabase...")
    try:
        from database.db_connect import test_connection
        if not test_connection():
            logger.warning("⚠️  Supabase connection ล้มเหลว - ตรวจสอบ config.py")
            print("⚠️  ไม่สามารถเชื่อมต่อ Supabase ได้ กรุณาตรวจสอบ SUPABASE_URL และ SUPABASE_KEY")
    except Exception as e:
        logger.warning(f"Supabase error: {e}")

    # เปิด GUI
    logger.info("เริ่มต้น GUI...")
    from gui.app import launch
    launch()


if __name__ == "__main__":
    main()
