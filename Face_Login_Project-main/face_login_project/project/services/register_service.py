# services/register_service.py - Register new user

import cv2
import os
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from core.encoding import encode_user_images, encode_image_file
from database.user_model import create_user, user_exists
from config import DATASET_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)


def draw_ui_overlay(frame: np.ndarray, username: str, img_count: int, num_images: int) -> np.ndarray:
    """
    Draw a clean UI overlay on the camera frame using PIL for Unicode support.
    """
    h, w = frame.shape[:2]

    # --- Dark gradient overlay at top ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # --- Dark overlay at bottom for progress bar ---
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    # --- Convert to PIL for Unicode text ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font (fallback to default if not found)
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
        font_sub   = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
        font_small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title
        font_small = font_title

    # --- Top bar text ---
    draw.text((16, 10), "Face Registration", font=font_title, fill=(255, 255, 255))
    draw.text((16, 38), f"User: {username}  •  Press SPACE to capture  •  ESC to cancel",
              font=font_small, fill=(180, 180, 180))

    # --- Progress bar at bottom ---
    bar_x, bar_y, bar_w, bar_h = 20, h - 40, w - 40, 14
    # Background track
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                            radius=7, fill=(60, 60, 60))
    # Filled portion
    filled_w = int(bar_w * (img_count / num_images))
    if filled_w > 0:
        draw.rounded_rectangle([bar_x, bar_y, bar_x + filled_w, bar_y + bar_h],
                                radius=7, fill=(0, 200, 120))

    # Progress label
    draw.text((bar_x, bar_y - 22),
              f"Captured: {img_count} / {num_images}",
              font=font_sub, fill=(200, 200, 200))

    # --- Face guide circle in center ---
    cx, cy, cr = w // 2, h // 2, min(w, h) // 4
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                 outline=(0, 220, 120), width=2)

    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def capture_user_images(username: str, num_images: int = 5) -> list:
    """
    Capture user images from camera and save to dataset/
    """
    user_dir = Path(DATASET_PATH) / username
    user_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    saved_paths = []
    img_count = 0
    flash_frames = 0  # White flash effect counter

    logger.info(f"Starting capture for '{username}' (SPACE to capture, ESC to cancel)")

    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw UI
        display = draw_ui_overlay(frame.copy(), username, img_count, num_images)

        # White flash effect on capture
        if flash_frames > 0:
            flash_overlay = display.copy()
            alpha = flash_frames / 8.0
            cv2.rectangle(flash_overlay, (0, 0), (display.shape[1], display.shape[0]),
                          (255, 255, 255), -1)
            cv2.addWeighted(flash_overlay, alpha, display, 1 - alpha, 0, display)
            flash_frames -= 1

        cv2.imshow("Register - Capture Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = str(user_dir / f"{username}_{img_count + 1}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)
            img_count += 1
            flash_frames = 8  # Trigger flash
            logger.info(f"Saved image: {path}")
        elif key == 27:  # ESC
            logger.info("Image capture cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_paths


def register_user(username: str, full_name: str, image_paths: list = None) -> dict:
    """
    Register a new user.
    Args:
        username: Username (must be unique)
        full_name: Full name
        image_paths: List of image file paths (optional, captures from camera if None)
    Returns: {"success": bool, "message": str, "user": dict | None}
    """
    if user_exists(username):
        return {"success": False, "message": f"Username '{username}' is already taken", "user": None}

    if not image_paths:
        logger.info(f"Starting image capture for {username}")
        image_paths = capture_user_images(username)

    if not image_paths:
        return {"success": False, "message": "No images available for registration", "user": None}

    user_dir = Path(DATASET_PATH) / username
    user_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        if not path.startswith(str(user_dir)):
            import shutil
            dest = str(user_dir / f"{username}_{i+1}.jpg")
            shutil.copy(path, dest)

    encoding = encode_user_images(username)
    if encoding is None:
        return {
            "success": False,
            "message": "Failed to encode face. Please use an image with a clearly visible face.",
            "user": None
        }

    user_record = create_user(username, full_name, encoding)
    if user_record:
        logger.info(f"Successfully registered '{username}'")
        return {
            "success": True,
            "message": f"Successfully registered '{full_name}'!",
            "user": user_record
        }
    else:
        return {
            "success": False,
            "message": "Failed to save data to database",
            "user": None
        }