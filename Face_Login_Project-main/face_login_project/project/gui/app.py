# gui/app.py - Main GUI (Cyberpunk Dark Theme)

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import queue
import time

from services.login_service import login_with_face, LoginResult
from services.register_service import register_user
from services.train_service import train_all_users
from database.log_model import get_logs, get_login_stats

logger = logging.getLogger(__name__)

# ─────────────────────────── Color Palette ───────────────────────────
BG_DEEP    = "#0b0e1a"
BG_PANEL   = "#0e1120"
BG_HEADER  = "#0d1122"
BG_CARD    = "#06080f"
BG_METRIC  = "#0d1122"

ACCENT     = "#3dffc0"
ACCENT_DIM = "#4ecfa0"
ACCENT_MUT = "#8eecd4"

TEXT_PRI   = "#e8e6f0"
TEXT_SEC   = "#8a8aa0"
TEXT_DIM   = "#445566"

BLUE       = "#5ca8f5"
AMBER      = "#f5c06a"
RED        = "#ff6b6b"
PURPLE     = "#b09ae8"

BORDER     = "#1a2035"
BORDER_ACC = "#1e3a2a"


class FaceLoginApp(tk.Tk):
    """Main window — Cyberpunk Dark Theme"""

    def __init__(self):
        super().__init__()
        self.title("FACE RECOGNITION LOGIN SYSTEM")
        self.geometry("940x700")
        self.resizable(False, False)
        self.configure(bg=BG_DEEP)

        self.camera_running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_frame = None
        self._scan_y = 0
        self._scan_dir = 1
        self._last_frame_time = time.time()  # ✅ for FPS calculation

        self._setup_fonts()
        self._build_ui()
        self._start_clock()
        self._start_camera_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────── Fonts & Styles ───────────────────────────

    def _setup_fonts(self):
        self.font_title  = ("Courier New", 13, "bold")
        self.font_mono   = ("Courier New", 9)
        self.font_mono_s = ("Courier New", 8)
        self.font_label  = ("Courier New", 8, "bold")
        self.font_metric = ("Courier New", 20, "bold")
        self.font_btn    = ("Courier New", 10, "bold")
        self.font_result = ("Courier New", 10)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=BG_CARD,
                        foreground=TEXT_SEC,
                        fieldbackground=BG_CARD,
                        borderwidth=0,
                        rowheight=26,
                        font=self.font_mono)
        style.configure("Dark.Treeview.Heading",
                        background=BG_PANEL,
                        foreground=TEXT_DIM,
                        borderwidth=0,
                        font=self.font_label,
                        relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", "#1a2540")],
                  foreground=[("selected", ACCENT)])
        style.layout("Dark.Treeview", [("Dark.Treeview.treearea", {"sticky": "nswe"})])

    # ─────────────────────────── UI Build ───────────────────────────

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=BG_DEEP)
        body.pack(fill="both", expand=True)
        self._build_left(body)
        self._build_right(body)
        self._build_footer()

    def _build_header(self):
        h = tk.Frame(self, bg=BG_HEADER, height=56)
        h.pack(fill="x")
        h.pack_propagate(False)

        inner = tk.Frame(h, bg=BG_HEADER)
        inner.pack(side="left", padx=20, pady=10)

        dot = tk.Canvas(inner, width=12, height=12, bg=BG_HEADER,
                        highlightthickness=0)
        dot.create_oval(1, 1, 11, 11, fill=ACCENT, outline="")
        dot.pack(side="left", padx=(0, 10))

        title_f = tk.Frame(inner, bg=BG_HEADER)
        title_f.pack(side="left")
        tk.Label(title_f, text="FACE RECOGNITION LOGIN SYSTEM",
                 font=("Courier New", 12, "bold"),
                 bg=BG_HEADER, fg="white").pack(anchor="w")
        tk.Label(title_f, text="Blink Detection Anti-Spoofing",
                 font=self.font_mono_s, bg=BG_HEADER, fg=ACCENT_DIM).pack(anchor="w")

        badge = tk.Label(h, text="  SYSTEM ONLINE  ",
                         font=self.font_label,
                         bg=BG_HEADER, fg=ACCENT,
                         relief="flat", bd=0, padx=8, pady=4)
        badge.pack(side="right", padx=20)
        badge.configure(highlightbackground=BORDER_ACC, highlightthickness=1)

    def _build_left(self, parent):
        lf = tk.Frame(parent, bg=BG_DEEP, width=500)
        lf.pack(side="left", fill="y", padx=(12, 6), pady=12)
        lf.pack_propagate(False)

        tk.Label(lf, text="// LIVE PREVIEW",
                 font=self.font_label, bg=BG_DEEP, fg=ACCENT).pack(anchor="w", pady=(0, 6))

        cam_wrap = tk.Frame(lf, bg=BORDER_ACC, padx=1, pady=1)
        cam_wrap.pack()

        self.canvas = tk.Canvas(cam_wrap, width=476, height=358,
                                bg=BG_CARD, highlightthickness=0)
        self.canvas.pack()
        self._draw_camera_overlay()

        sb = tk.Frame(lf, bg="#0a1a12", highlightbackground=BORDER_ACC,
                      highlightthickness=1)
        sb.pack(fill="x", pady=(8, 0))

        self._status_dot = tk.Canvas(sb, width=10, height=10,
                                     bg="#0a1a12", highlightthickness=0)
        self._status_dot.create_oval(1, 1, 9, 9, fill=ACCENT, tags="dot", outline="")
        self._status_dot.pack(side="left", padx=(10, 6), pady=8)

        self.status_var = tk.StringVar(value="Ready — waiting for face detection...")
        tk.Label(sb, textvariable=self.status_var,
                 font=self.font_mono, bg="#0a1a12", fg=ACCENT_MUT,
                 anchor="w").pack(side="left", fill="x", expand=True,
                                  padx=(0, 10), pady=8)
        self._pulse_dot()

        mf = tk.Frame(lf, bg=BG_DEEP)
        mf.pack(fill="x", pady=(8, 0))

        metrics = [("FPS",    "0",  ACCENT, "fps"),
                   ("BLINKS", "0",  AMBER,  "blink"),
                   ("FACES",  "0",  BLUE,   "face")]

        self._metric_vars = {}
        for label, val, color, key in metrics:
            card = tk.Frame(mf, bg=BG_METRIC,
                            highlightbackground=BORDER, highlightthickness=1)
            card.pack(side="left", fill="x", expand=True, padx=3)
            tk.Label(card, text=label, font=self.font_label,
                     bg=BG_METRIC, fg=TEXT_DIM).pack(anchor="w", padx=10, pady=(8, 2))
            v = tk.StringVar(value=val)
            self._metric_vars[key] = v
            tk.Label(card, textvariable=v, font=self.font_metric,
                     bg=BG_METRIC, fg=color).pack(anchor="w", padx=10, pady=(0, 8))

    def _draw_camera_overlay(self):
        c = self.canvas
        for x in range(0, 477, 40):
            c.create_line(x, 0, x, 358, fill="#0e1a12", width=1, tags="grid")
        for y in range(0, 359, 40):
            c.create_line(0, y, 476, y, fill="#0e1a12", width=1, tags="grid")

        cx, cy = 238, 170
        c.create_oval(cx-80, cy-95, cx+80, cy+95,
                      outline=ACCENT, width=1, dash=(4, 6), fill="", tags="face_oval")
        c.create_oval(cx-35, cy-25, cx-20, cy-12, fill=ACCENT, outline="", tags="eye")
        c.create_oval(cx+20, cy-25, cx+35, cy-12, fill=ACCENT, outline="", tags="eye")
        c.create_arc(cx-28, cy+15, cx+28, cy+45, start=200, extent=140,
                     style="arc", outline=ACCENT, width=1, tags="mouth")

        s, t = 28, 2
        corners = [(cx-90, cy-100, 1, 1), (cx+90, cy-100, -1, 1),
                   (cx-90, cy+100, 1, -1), (cx+90, cy+100, -1, -1)]
        for bx, by, dx, dy in corners:
            c.create_line(bx, by, bx + dx*s, by, fill=ACCENT, width=t)
            c.create_line(bx, by, bx, by + dy*s, fill=ACCENT, width=t)

        self._scan_line = c.create_line(cx-90, cy-100, cx+90, cy-100,
                                        fill=ACCENT, width=2, tags="scan")
        self._animate_scan()

    def _animate_scan(self):
        if not self.winfo_exists():
            return
        try:
            coords = self.canvas.coords(self._scan_line)
            if not coords:
                return
            y = coords[1] + self._scan_dir * 3
            if y >= 270: self._scan_dir = -1
            elif y <= 75: self._scan_dir = 1
            self.canvas.coords(self._scan_line, coords[0], y, coords[2], y)
        except tk.TclError:
            return
        self.after(40, self._animate_scan)

    def _build_right(self, parent):
        rf = tk.Frame(parent, bg=BG_PANEL)
        rf.pack(side="right", fill="both", expand=True, padx=(6, 12), pady=12)
        self._build_result_box(rf)
        self._build_buttons(rf)
        self._build_log_table(rf)

    def _build_result_box(self, parent):
        tk.Label(parent, text="// DETECTION RESULT",
                 font=self.font_label, bg=BG_PANEL, fg=ACCENT).pack(anchor="w", pady=(0, 4))

        rb = tk.Frame(parent, bg=BG_CARD,
                      highlightbackground=BORDER_ACC, highlightthickness=1)
        rb.pack(fill="x", pady=(0, 10))

        fields = [
            ("STATUS",     "READY", ACCENT,   "r_status"),
            ("USER",       "—",     TEXT_PRI, "r_user"),
            ("CONFIDENCE", "—",     ACCENT,   "r_conf"),
            ("TIME",       "—",     TEXT_PRI, "r_time"),
            ("ANTI-SPOOF", "PASS",  ACCENT,   "r_spoof"),
        ]
        self._result_vars = {}
        for i, (key, val, color, attr) in enumerate(fields):
            row = tk.Frame(rb, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=5)
            if i < len(fields) - 1:
                tk.Frame(rb, bg=BORDER, height=1).pack(fill="x", padx=14)

            tk.Label(row, text=key, font=self.font_label,
                     bg=BG_CARD, fg=TEXT_DIM, width=12, anchor="w").pack(side="left")
            v = tk.StringVar(value=val)
            self._result_vars[attr] = v
            lbl = tk.Label(row, textvariable=v, font=self.font_result,
                           bg=BG_CARD, fg=color, anchor="e")
            lbl.pack(side="right")
            setattr(self, f"_lbl_{attr}", lbl)

        bar_row = tk.Frame(rb, bg=BG_CARD)
        bar_row.pack(fill="x", padx=14, pady=(0, 10))
        self._conf_bar_bg = tk.Frame(bar_row, bg=BORDER, height=4)
        self._conf_bar_bg.pack(fill="x")
        self._conf_bar_fg = tk.Frame(self._conf_bar_bg, bg=ACCENT, height=4)
        self._conf_bar_fg.place(relx=0, rely=0, relwidth=0, relheight=1)

    def _build_buttons(self, parent):
        bf = tk.Frame(parent, bg=BG_PANEL)
        bf.pack(fill="x", pady=(0, 10))

        tk.Button(bf, text="▶  LOGIN",
                  font=("Courier New", 12, "bold"),
                  bg=ACCENT, fg="#042c1a",
                  activebackground="#2de8aa", activeforeground="#042c1a",
                  relief="flat", bd=0, cursor="hand2",
                  command=self._start_login
                  ).pack(fill="x", pady=(0, 6), ipady=8)

        sub_frame = tk.Frame(bf, bg=BG_PANEL)
        sub_frame.pack(fill="x")

        for text, bg, fg, cmd in [
            ("📝 Register",   "#1a3a6e", BLUE,   self._open_register),
            ("⚙  Re-Train",   "#2a1a4e", PURPLE, self._retrain),
            ("📊 Statistics", "#2e1a10", AMBER,  self._show_stats),
        ]:
            tk.Button(sub_frame, text=text, font=self.font_btn,
                      bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                      relief="flat", bd=0, cursor="hand2", command=cmd
                      ).pack(side="left", fill="x", expand=True, padx=3, ipady=7)

    def _build_log_table(self, parent):
        tk.Label(parent, text="// RECENT LOGS",
                 font=self.font_label, bg=BG_PANEL, fg=TEXT_DIM).pack(anchor="w", pady=(4, 4))

        cols = ("Time", "User", "Status", "Conf.")
        self.log_tree = ttk.Treeview(parent, columns=cols,
                                      show="headings", height=6,
                                      style="Dark.Treeview")
        widths = {"Time": 130, "User": 100, "Status": 90, "Conf.": 65}
        for col in cols:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=widths[col], anchor="center")

        self.log_tree.tag_configure("ok",    foreground=ACCENT)
        self.log_tree.tag_configure("fail",  foreground=RED)
        self.log_tree.tag_configure("spoof", foreground=AMBER)

        self.log_tree.pack(fill="both", expand=True)
        self._refresh_logs()

    # ─────────────────────────── Clock ───────────────────────────

    def _start_clock(self):
        self._clock_var = tk.StringVar()
        self._tick()

    def _tick(self):
        if not self.winfo_exists():
            return
        self._clock_var.set(time.strftime("%H:%M:%S"))
        self.after(1000, self._tick)

    # ─────────────────────────── Camera ───────────────────────────

    def _pause_camera(self):
        """Stop preview camera thread and release capture device."""
        self.camera_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _start_camera_preview(self):
        if self.camera_running:
            return

        try:
            from config import CAMERA_INDEX
        except ImportError:
            CAMERA_INDEX = 0

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self._update_status("Unable to open camera for preview")
            return

        self.camera_running = True
        self._metric_vars["fps"].set("0")
        self._metric_vars["face"].set("0")
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self._update_canvas()

    def _camera_loop(self):
        """Camera loop — also updates FPS and FACES metrics during preview."""
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # ✅ FPS calculation
            now = time.time()
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                fps = int(1.0 / elapsed)
                self.after(0, self._metric_vars["fps"].set, str(min(fps, 99)))
            self._last_frame_time = now

            # ✅ Face count during idle preview
            try:
                from core.face_recognition_core import detect_faces
                faces = detect_faces(frame)
                self.after(0, self._metric_vars["face"].set, str(len(faces)))
            except Exception:
                pass

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def _update_canvas(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get_nowait()
            self.current_frame = frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((476, 358))
            photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo
            self._redraw_corners()
        self.after(30, self._update_canvas)

    def _redraw_corners(self):
        self.canvas.delete("corner")
        cx, cy, s, t = 238, 170, 28, 2
        for bx, by, dx, dy in [(cx-90, cy-100, 1, 1), (cx+90, cy-100, -1, 1),
                                (cx-90, cy+100, 1, -1), (cx+90, cy+100, -1, -1)]:
            self.canvas.create_line(bx, by, bx+dx*s, by, fill=ACCENT, width=t, tags="corner")
            self.canvas.create_line(bx, by, bx, by+dy*s, fill=ACCENT, width=t, tags="corner")

    # ─────────────────────────── Status / Result ───────────────────────────

    def _pulse_dot(self):
        if not self.winfo_exists():
            return
        self._pulse_state = getattr(self, "_pulse_state", True)
        color = ACCENT if self._pulse_state else BG_CARD
        try:
            self._status_dot.itemconfig("dot", fill=color)
        except tk.TclError:
            return
        self._pulse_state = not self._pulse_state
        self.after(900, self._pulse_dot)

    def _update_status(self, msg):
        self.status_var.set(msg)

    def _set_result(self, status, user, conf, elapsed, spoof_ok, success):
        color_s  = ACCENT if success else RED
        color_sp = ACCENT if spoof_ok else RED

        self._result_vars["r_status"].set(status)
        self._lbl_r_status.configure(fg=color_s)
        self._result_vars["r_user"].set(user)
        self._result_vars["r_conf"].set(conf)
        self._result_vars["r_time"].set(elapsed)
        self._result_vars["r_spoof"].set("PASS" if spoof_ok else "FAIL")
        self._lbl_r_spoof.configure(fg=color_sp)

        try:
            pct = float(conf.replace("%", "")) / 100
        except Exception:
            pct = 0
        self._conf_bar_fg.place(relwidth=pct)

    def _show_result(self, result: LoginResult):
        conf_str    = f"{result.confidence * 100:.1f}%"
        elapsed_str = f"{result.elapsed_time:.2f}s"
        spoof_ok    = result.status != "spoof_detected"

        self._set_result(
            status="SUCCESS" if result.success else "FAILED",
            user=result.username or "Unknown",
            conf=conf_str,
            elapsed=elapsed_str,
            spoof_ok=spoof_ok,
            success=result.success,
        )
        self._refresh_logs()

    # ─────────────────────────── Actions ───────────────────────────

    def _start_login(self):
        self._update_status("Starting login process...")
        self._set_result("SCANNING", "—", "—", "—", True, True)
        self._lbl_r_status.configure(fg=AMBER)

        # Pause camera preview before login takes over
        self._pause_camera()

        def do_login():
            result = login_with_face(
                callback_frame=self._on_login_frame,   # ✅ live frame during login
                callback_status=lambda m: self.after(0, self._update_status, m)
            )
            self.after(0, self._on_login_done, result)

        threading.Thread(target=do_login, daemon=True).start()

    def _on_login_frame(self, frame):
        """
        Called by login_service for each frame during login.
        Updates canvas + FPS + FACES metrics in real time.
        """
        # Push frame to canvas
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

        # ✅ FPS
        now     = time.time()
        elapsed = now - self._last_frame_time
        if elapsed > 0:
            fps = int(1.0 / elapsed)
            self.after(0, self._metric_vars["fps"].set, str(min(fps, 99)))
        self._last_frame_time = now

        # ✅ FACES
        try:
            from core.face_recognition_core import detect_faces
            faces = detect_faces(frame)
            self.after(0, self._metric_vars["face"].set, str(len(faces)))
        except Exception:
            pass

    def _on_login_done(self, result: LoginResult):
        """Called on main thread when login_with_face() returns."""
        self._show_result(result)

        # ✅ Update BLINKS metric from real result
        self._metric_vars["blink"].set(str(result.blink_count))

        if result.success:
            messagebox.showinfo("✅ Login Successful", result.message)
        else:
            messagebox.showwarning("❌ Login Failed", result.message)

        # Restart preview camera
        self._start_camera_preview()

    def _open_register(self):
        reg_win = tk.Toplevel(self)
        reg_win.title("Register New User")
        reg_win.geometry("380x260")
        reg_win.configure(bg=BG_DEEP)
        reg_win.resizable(False, False)
        reg_win.grab_set()

        tk.Label(reg_win, text="// REGISTER NEW USER",
                 font=("Courier New", 11, "bold"),
                 bg=BG_DEEP, fg=ACCENT).pack(pady=(18, 12))

        for label, attr in [("USERNAME", "uname"), ("FULL NAME", "fname")]:
            row = tk.Frame(reg_win, bg=BG_DEEP)
            row.pack(fill="x", padx=30, pady=4)
            tk.Label(row, text=label, font=self.font_label,
                     bg=BG_DEEP, fg=TEXT_DIM, anchor="w").pack(fill="x")
            e = tk.Entry(row, font=self.font_result,
                         bg=BG_METRIC, fg=TEXT_PRI,
                         insertbackground=ACCENT,
                         relief="flat", bd=0,
                         highlightbackground=BORDER_ACC,
                         highlightthickness=1)
            e.pack(fill="x", ipady=6)
            setattr(reg_win, attr, e)

        msg_lbl = tk.Label(reg_win, text="", font=self.font_mono_s,
                           bg=BG_DEEP, fg=AMBER)
        msg_lbl.pack()

        def do_register():
            username  = reg_win.uname.get().strip()
            full_name = reg_win.fname.get().strip()
            if not username or not full_name:
                msg_lbl.config(text="Please fill in all fields")
                return

            # Pause main camera preview before register capture opens
            self._pause_camera()

            reg_win.destroy()
            self._update_status(f"Registering '{username}'...")

            def run():
                result = register_user(username, full_name)
                msg = result["message"]
                self.after(0, self._update_status, msg)
                if result["success"]:
                    self.after(0, messagebox.showinfo,  "✅ Success", msg)
                else:
                    self.after(0, messagebox.showerror, "❌ Error",   msg)

                # Resume camera preview after registration completes
                self.after(0, self._start_camera_preview)

            threading.Thread(target=run, daemon=True).start()

        tk.Button(reg_win, text="▶  Capture & Register",
                  font=("Courier New", 10, "bold"),
                  bg=ACCENT, fg="#042c1a",
                  activebackground="#2de8aa",
                  relief="flat", bd=0, cursor="hand2",
                  command=do_register).pack(fill="x", padx=30, pady=(8, 0), ipady=8)

    def _retrain(self):
        if messagebox.askyesno("Re-Train", "Re-train all encodings. Proceed?"):
            self._update_status("Re-training encodings...")

            def run():
                result = train_all_users()
                msg = f"Training complete: {result['success']}/{result['total']} succeeded"
                self.after(0, self._update_status, msg)
                self.after(0, messagebox.showinfo, "✅ Training Done", msg)

            threading.Thread(target=run, daemon=True).start()

    def _show_stats(self):
        stats = get_login_stats()
        msg = (
            f"LOGIN STATISTICS\n"
            f"{'─'*30}\n"
            f"Total Logins : {stats['total']}\n"
            f"SUCCESS      : {stats['success']}\n"
            f"FAILED       : {stats['failed']}\n"
            f"SPOOF DETECT : {stats['spoof_detected']}\n"
            f"SUCCESS RATE : {stats['success_rate']:.1f}%"
        )
        messagebox.showinfo("Statistics", msg)

    def _refresh_logs(self):
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        logs = get_logs(limit=10)
        for log in logs:
            ts     = log.get("created_at_local", log.get("created_at", ""))[:19]
            user   = log.get("username") or "Unknown"
            status = log.get("status", "")
            conf   = f"{log.get('confidence', 0) * 100:.0f}%"

            if status == "success":
                tag, status_disp = "ok",    "SUCCESS"
            elif status == "spoof_detected":
                tag, status_disp = "spoof", "SPOOF"
            else:
                tag, status_disp = "fail",  "FAILED"

            self.log_tree.insert("", "end",
                                 values=(ts, user, status_disp, conf),
                                 tags=(tag,))

    # ─────────────────────────── Footer ───────────────────────────

    def _build_footer(self):
        f = tk.Frame(self, bg=BG_HEADER, height=28)
        f.pack(fill="x", side="bottom")
        f.pack_propagate(False)

        dot = tk.Canvas(f, width=8, height=8, bg=BG_HEADER, highlightthickness=0)
        dot.create_oval(1, 1, 7, 7, fill=ACCENT, outline="")
        dot.pack(side="left", padx=(16, 6), pady=10)

        tk.Label(f, text="Powered by face_recognition + dlib + Supabase",
                 font=("Courier New", 8), bg=BG_HEADER, fg=TEXT_DIM).pack(side="left")

        self._start_clock()
        tk.Label(f, textvariable=self._clock_var,
                 font=("Courier New", 8, "bold"),
                 bg=BG_HEADER, fg=ACCENT).pack(side="right", padx=16)

    # ─────────────────────────── Cleanup ───────────────────────────

    def _on_close(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()


def launch():
    app = FaceLoginApp()
    app.mainloop()