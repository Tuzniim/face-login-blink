# analytics/accuracy_plot.py - วาดกราฟวิเคราะห์จาก logs

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import Counter
import logging

from database.log_model import get_logs, get_login_stats

logger = logging.getLogger(__name__)


def plot_login_history(limit: int = 100, save_path: str = None):
    """
    กราฟ Login Success vs Failed ตามเวลา
    """
    logs = get_logs(limit=limit)
    if not logs:
        print("ไม่มีข้อมูล log")
        return

    dates, statuses = [], []
    for log in reversed(logs):
        try:
            dt = datetime.fromisoformat(log["created_at"].replace("Z", ""))
            dates.append(dt)
            statuses.append(log["status"])
        except Exception:
            continue

    success_times = [d for d, s in zip(dates, statuses) if s == "success"]
    failed_times = [d for d, s in zip(dates, statuses) if s != "success"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(success_times, ["success"] * len(success_times),
               color="#00b894", marker="o", s=80, label="✅ Success", zorder=3)
    ax.scatter(failed_times, ["failed"] * len(failed_times),
               color="#d63031", marker="x", s=80, label="❌ Failed/Spoof", zorder=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha="right")

    ax.set_title("Login History Timeline", fontsize=14, fontweight="bold")
    ax.set_ylabel("Status")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"บันทึกกราฟที่ {save_path}")
    else:
        plt.show()
    plt.close()


def plot_status_pie(save_path: str = None):
    """
    Pie chart สัดส่วน login status
    """
    stats = get_login_stats()
    if stats["total"] == 0:
        print("ไม่มีข้อมูล")
        return

    labels = ["✅ Success", "❌ Failed", "🚨 Spoof"]
    values = [stats["success"], stats["failed"], stats["spoof_detected"]]
    colors = ["#00b894", "#d63031", "#e17055"]

    # กรองที่มีค่า > 0
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return
    labels, values, colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.set_title(
        f"Login Status Distribution\n(Total: {stats['total']} logins)",
        fontsize=13, fontweight="bold"
    )

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_confidence_distribution(save_path: str = None):
    """
    Histogram การกระจาย confidence score
    """
    logs = get_logs(limit=500)
    success_conf = [l["confidence"] for l in logs if l["status"] == "success" and l.get("confidence")]
    failed_conf = [l["confidence"] for l in logs if l["status"] != "success" and l.get("confidence")]

    if not success_conf and not failed_conf:
        print("ไม่มีข้อมูล confidence")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    if success_conf:
        ax.hist(success_conf, bins=20, color="#00b894", alpha=0.7, label="✅ Success", edgecolor="white")
    if failed_conf:
        ax.hist(failed_conf, bins=20, color="#d63031", alpha=0.7, label="❌ Failed", edgecolor="white")

    ax.axvline(x=0.5, color="#fdcb6e", linestyle="--", linewidth=2, label="Threshold (0.5)")
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Score Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def generate_all_plots(output_dir: str = "analytics/output"):
    """สร้างกราฟทั้งหมด"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_login_history(save_path=f"{output_dir}/login_history.png")
    plot_status_pie(save_path=f"{output_dir}/status_pie.png")
    plot_confidence_distribution(save_path=f"{output_dir}/confidence_dist.png")
    print(f"✅ สร้างกราฟทั้งหมดที่ {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
