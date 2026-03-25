# database/log_model.py - Login log recording

import logging
from datetime import datetime, timezone, timedelta
from database.db_connect import get_client

logger = logging.getLogger(__name__)

# ---- Supabase Table Schema ----
# CREATE TABLE login_logs (
#   id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#   username TEXT,
#   status TEXT NOT NULL,          -- 'success' | 'failed' | 'spoof_detected'
#   confidence FLOAT,              -- accuracy score (1 - distance)
#   blink_count INTEGER,           -- number of blinks detected
#   message TEXT,
#   ip_address TEXT,
#   created_at TIMESTAMPTZ DEFAULT NOW()
# );

TABLE_NAME = "login_logs"

TH_TZ = timezone(timedelta(hours=7))  # ✅ Thailand timezone (UTC+7)


def now_th() -> datetime:
    """Returns current datetime in Thailand timezone (UTC+7)."""
    return datetime.now(tz=TH_TZ)


def format_th_time(dt_str: str) -> str:
    """
    Convert UTC datetime string from Supabase → Thailand local time string.
    Input:  '2026-03-25T05:43:01.123+00:00'
    Output: '2026-03-25 12:43:01'
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # assume UTC if no tz info
        dt_local = dt.astimezone(TH_TZ)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_str  # return original string if parsing fails


def insert_log(
    username: str | None,
    status: str,
    confidence: float = 0.0,
    blink_count: int = 0,
    message: str = ""
) -> dict | None:
    """
    Save a login log to Supabase.
    Args:
        username:   Username (None if unknown)
        status:     'success' | 'failed' | 'spoof_detected' | 'timeout'
        confidence: Accuracy score 0.0–1.0
        blink_count: Number of blinks detected
        message:    Additional message
    Returns: log record or None
    """
    try:
        client = get_client()
        data = {
            "username": username,
            "status": status,
            "confidence": round(confidence, 4),
            "blink_count": blink_count,
            "message": message,
            "created_at": now_th().isoformat()  # ✅ UTC+7 with timezone info
        }
        response = client.table(TABLE_NAME).insert(data).execute()
        if response.data:
            logger.info(f"Log saved: user={username}, status={status}")
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to save log: {e}")
        return None


def get_logs(limit: int = 100, username: str = None) -> list:
    """
    Fetch login logs.
    Args:
        limit:    Maximum number of records
        username: Filter by username (optional)
    Returns: list of log records (with 'created_at_local' field added)
    """
    try:
        client = get_client()
        query = (
            client.table(TABLE_NAME)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if username:
            query = query.eq("username", username)

        response = query.execute()
        logs = response.data or []

        # ✅ Add local time field for display in UI
        for log in logs:
            log["created_at_local"] = format_th_time(log.get("created_at", ""))

        return logs
    except Exception as e:
        logger.error(f"Failed to fetch logs: {e}")
        return []


def get_login_stats(username: str = None) -> dict:
    """
    Summarize login statistics.
    Returns: {
        "total": int,
        "success": int,
        "failed": int,
        "spoof_detected": int,
        "success_rate": float
    }
    """
    logs = get_logs(limit=1000, username=username)
    total = len(logs)
    success = sum(1 for l in logs if l["status"] == "success")
    failed  = sum(1 for l in logs if l["status"] == "failed")
    spoof   = sum(1 for l in logs if l["status"] == "spoof_detected")

    return {
        "total":        total,
        "success":      success,
        "failed":       failed,
        "spoof_detected": spoof,
        "success_rate": (success / total * 100) if total > 0 else 0.0
    }