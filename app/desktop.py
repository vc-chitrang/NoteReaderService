"""
Desktop developer console for the note scanner (PySide6).

    .venv\\Scripts\\python -m app.desktop        (or double-click start.bat)

Shows the live camera with the note box and scan state, what the wall would show, every read
(crop, name, deed, language, timings, raw model answer) and the live log. Each read is play-tested
here first: Approve / Reject records whether the result is right. Nothing is sent to the wall app
yet - the HTTP API keeps running in the background for when it is hooked up.

Photos and visitors' words stay in memory only; the log file never contains them.
"""
import json
import sys
import threading
import time

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget)

from . import __version__, engine
from .detector import normalize_image

cfg, hub, camera, scanner = engine.cfg, engine.hub, engine.camera, engine.scanner

KIND_COLOURS = {"info": "#ffe829", "success": "#a0eb96", "warning": "#ffaa5a", None: "#9aa0a6"}
REVIEW_TEXT = {None: "not reviewed", "approved": "approved — would send to wall", "rejected": "rejected"}
LOG_COLOURS = {"error": "#ff7b72", "warn": "#ffaa5a", "info": "#c9d1d9", "debug": "#8b949e"}

STYLE = """
QWidget { background: #14161a; color: #d7dbe0; font-family: 'Segoe UI'; font-size: 10pt; }
QGroupBox { border: 1px solid #2b3038; border-radius: 6px; margin-top: 14px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8b949e; }
QPushButton { background: #232830; border: 1px solid #353b45; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #2c323c; }
QPushButton:checked { background: #6b4e16; border-color: #a77b22; }
QPushButton:disabled { color: #5c636d; }
QPushButton#approve { background: #1f3d24; border-color: #2f6b38; }
QPushButton#reject { background: #4a1f1f; border-color: #7a3030; }
QComboBox, QPlainTextEdit, QListWidget { background: #0d0f12; border: 1px solid #2b3038; border-radius: 4px; }
QPlainTextEdit { font-family: 'Cascadia Mono', 'Consolas'; font-size: 9pt; }
QLabel#banner { border-radius: 6px; padding: 10px 14px; font-size: 14pt; }
QLabel#field { font-size: 12pt; color: #ffffff; }
QLabel#muted { color: #8b949e; }
QSplitter::handle { background: #1d2127; }
"""


# ---------------------------------------------------------------------- background helpers

class EventPump(QThread):
    """Pulls engine events (log lines, state changes, read results) and hands them to the UI thread."""
    events = Signal(list)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        seq = 0
        while self._running:
            items = hub.since(seq, timeout=0.5)
            if items:
                seq = items[-1]["seq"]
                self.events.emit(items)

    def stop(self):
        self._running = False
        self.wait(2000)


class Worker(QObject):
    """Runs a blocking call off the UI thread and reports back."""
    finished = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def start(self):
        def run():
            try:
                value = self._fn()
            except Exception as e:      # reported to the UI, never crashes it
                value = e
            self.finished.emit(value)
        threading.Thread(target=run, daemon=True).start()


def to_pixmap(frame_bgr):
    h, w = frame_bgr.shape[:2]
    image = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(image.copy())


# ---------------------------------------------------------------------- approval window

class ApprovalDialog(QDialog):
    """Debug gate for a traced note: original, black and white, cleaned, final neon strokes, side by side.
    Nothing goes to the wall until Approve is pressed."""

    def __init__(self, parent, capture_id, original_jpeg, stages, decide):
        super().__init__(parent)
        self.capture_id = capture_id
        self._decide = decide
        self._decided = False
        self.setWindowTitle("Check the note #%d before it goes to the wall" % capture_id)
        screen = QApplication.primaryScreen().availableGeometry()
        panel_w = max(260, min(430, (screen.width() - 120) // 4))

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        panels = [("Original photo", original_jpeg), ("1 · Black and white", stages["bw"]),
                  ("2 · Cleaned: holes filled, ruled lines removed", stages["clean"]),
                  ("3 · Strokes sent to the wall, over the cleaned image (name cyan, message yellow)", stages["final"])]
        for title, data in panels:
            box = QVBoxLayout()
            caption = QLabel(title)
            caption.setObjectName("muted")
            caption.setWordWrap(True)
            image = QLabel()
            image.setStyleSheet("background: #000;")
            pixmap = QPixmap()
            if data:
                pixmap.loadFromData(data)
                scaled = pixmap.scaledToWidth(panel_w, Qt.SmoothTransformation)
                image.setPixmap(scaled)
                image.setFixedSize(scaled.size())
            else:
                image.setText("(not kept)")
            image.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            box.addWidget(caption)
            box.addWidget(image, 0, Qt.AlignHCenter)
            box.addStretch(1)
            row.addLayout(box, 1)
        layout.addLayout(row, 1)

        buttons = QHBoxLayout()
        self.approve_button = QPushButton("Approve — send to the wall  (A)")
        self.approve_button.setObjectName("approve")
        self.reject_button = QPushButton("Reject — do not send  (X)")
        self.reject_button.setObjectName("reject")
        self.approve_button.clicked.connect(lambda: self.finish(True))
        self.reject_button.clicked.connect(lambda: self.finish(False))
        buttons.addStretch(1)
        buttons.addWidget(self.reject_button)
        buttons.addWidget(self.approve_button)
        layout.addLayout(buttons)
        QShortcut(QKeySequence("A"), self, activated=lambda: self.finish(True))
        QShortcut(QKeySequence("X"), self, activated=lambda: self.finish(False))
        self.resize(panel_w * 4 + 90, 520)

    def finish(self, approved):
        if self._decided:
            return
        self._decided = True
        self._decide(self.capture_id, approved)
        hub.log("Note #%d %s in the approval window." % (self.capture_id, "approved" if approved else "rejected"))
        self.close()

    def closeEvent(self, event):
        if not self._decided:       # closing the window without choosing means "do not send"
            self._decided = True
            self._decide(self.capture_id, False)
        super().closeEvent(event)


# ---------------------------------------------------------------------- window

class Console(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Good Deed Tree — Note Reader %s (developer console)" % __version__)
        self.resize(1600, 960)
        self.entries = {}         # capture id -> history entry (memory only)
        self.current_id = None
        self._workers = []

        root = QSplitter(Qt.Horizontal)
        root.addWidget(self._build_left())
        root.addWidget(self._build_right())
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 2)
        self.setCentralWidget(root)

        QShortcut(QKeySequence("C"), self, activated=self.capture)
        QShortcut(QKeySequence("R"), self, activated=self.relearn)
        QShortcut(QKeySequence("P"), self, activated=lambda: self.pause_button.toggle())

        self.video_timer = QTimer(self, interval=66, timeout=self.update_video)       # ~15 fps preview
        self.status_timer = QTimer(self, interval=1000, timeout=self.update_status)
        self.video_timer.start()
        self.status_timer.start()

        self.pump = EventPump()
        self.pump.events.connect(self.on_events)
        self.pump.start()

    # ------------------------------------------------------------------ layout

    def _build_left(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.video = QLabel("Waiting for the camera…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet("background:#000; border-radius:6px; color:#5c636d;")
        layout.addWidget(self.video, 1)

        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.set_banner(None)
        layout.addWidget(self.banner)

        controls = QHBoxLayout()
        self.capture_button = QPushButton("Capture  [C]", clicked=self.capture)
        self.relearn_button = QPushButton("Relearn desk  [R]", clicked=self.relearn)
        self.pause_button = QPushButton("Pause  [P]", checkable=True, toggled=self.pause)
        self.auto_box = QCheckBox("Auto scan", checked=scanner.auto_scan, toggled=self.auto_scan)
        self.review_box = QCheckBox("Review before sending", checked=scanner.cfg["handwriting"]["require_approval"],
                                    toggled=self.set_review_mode)
        self.test_button = QPushButton("Test image…", clicked=self.test_image)
        for w in (self.capture_button, self.relearn_button, self.pause_button, self.auto_box, self.review_box, self.test_button):
            controls.addWidget(w)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status_line = QLabel()
        self.status_line.setObjectName("muted")
        layout.addWidget(self.status_line)
        return panel

    def _build_right(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        card = QGroupBox("Last note")
        card_layout = QHBoxLayout(card)
        self.crop = QLabel("No note yet")
        self.crop.setAlignment(Qt.AlignCenter)
        self.crop.setFixedSize(260, 260)
        self.crop.setStyleSheet("background:#0d0f12; border-radius:4px; color:#5c636d;")
        card_layout.addWidget(self.crop)

        form = QFormLayout()
        self.fields = {}
        for key, label in (("outcome", "Outcome"), ("timings", "Timings")):
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(QLabel(label), value)
            self.fields[key] = value
        right = QVBoxLayout()
        right.addLayout(form)
        hint = QLabel("Notes are sent to the wall automatically. Tick \"Review before sending\" to check each one first.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        right.addWidget(hint)
        card_layout.addLayout(right, 1)
        layout.addWidget(card)

        lower = QSplitter(Qt.Vertical)
        history_box = QGroupBox("History (this session, memory only)")
        history_layout = QVBoxLayout(history_box)
        self.history = QListWidget()
        self.history.currentItemChanged.connect(self.on_history_selected)
        history_layout.addWidget(self.history)
        lower.addWidget(history_box)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log)
        lower.addWidget(log_box)
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 2)
        layout.addWidget(lower, 1)
        return panel

    # ------------------------------------------------------------------ live view

    def update_video(self):
        frame, _ = camera.latest()
        if frame is None:
            return
        view = engine.overlay(frame, width=1280)
        pixmap = to_pixmap(view)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self):
        s = scanner.status()
        cam = "%s %dx%d" % (camera.device_name, camera.size[0], camera.size[1]) if camera.ok else "no camera"
        self.status_line.setText(
            "Camera: %s   |   State: %s   |   Notes: %d, avg %.1f s   |   Queue: %d" % (
                cam, s["state"].replace("_", " "), s["reads"], s["avg_read_ms"] / 1000.0, s["queue"]))

    def set_banner(self, message):
        if not message:
            self.banner.setText("Wall would show: (nothing)")
            self.banner.setStyleSheet("background:#1d2127; color:#8b949e;")
            return
        colour = KIND_COLOURS.get(message.get("kind"), "#d7dbe0")
        text = "Wall would show:  <b>%s</b>" % message["title"]
        if message.get("sub"):
            text += "<br><span style='font-size:11pt;color:#c9d1d9'>%s</span>" % message["sub"]
        self.banner.setText(text)
        self.banner.setStyleSheet("background:#1d2127; color:%s; border-left:5px solid %s;" % (colour, colour))

    # ------------------------------------------------------------------ events

    def on_events(self, items):
        for item in items:
            kind, data = item["kind"], item.get("data") or {}
            if item.get("text"):
                self.append_log(item["text"], item.get("level", "info"), item["t"])
            if kind == "state":
                self.set_banner(data.get("message"))
            elif kind == "result":
                self.add_result(data)
            elif kind == "approval":
                self.open_approval(data.get("id"))

    def open_approval(self, capture_id):
        stages = scanner.previews.get(capture_id)
        if not stages:
            scanner.decide(capture_id, False)
            return
        dialog = ApprovalDialog(self, capture_id, scanner.captures.get(capture_id), stages, scanner.decide)
        self._approval = dialog        # keep a reference while it is open
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QApplication.alert(self)

    def append_log(self, text, level="info", t=None):
        stamp = time.strftime("%H:%M:%S", time.localtime(t or time.time()))
        colour = LOG_COLOURS.get(level, "#c9d1d9")
        self.log.appendHtml("<span style='color:#5c636d'>%s</span> <span style='color:%s'>%s</span>" % (
            stamp, colour, text.replace("&", "&amp;").replace("<", "&lt;")))
        self.log.moveCursor(QTextCursor.End)

    def add_result(self, entry):
        self.entries[entry["id"]] = entry
        code = entry["outcome"]["code"]
        item = QListWidgetItem(self.history_label(entry))
        item.setData(Qt.UserRole, entry["id"])
        item.setForeground(QColor(KIND_COLOURS.get(entry["outcome"]["kind"], "#d7dbe0")))
        self.history.insertItem(0, item)
        while self.history.count() > 50:
            old = self.history.takeItem(self.history.count() - 1)
            self.entries.pop(old.data(Qt.UserRole), None)
        self.history.setCurrentItem(item)      # shows it in the card
        if code == "accepted":
            QApplication.alert(self)

    def history_label(self, entry):
        return "#%d  %s  %-11s  %s" % (
            entry["id"], time.strftime("%H:%M:%S", time.localtime(entry["t"])), entry["source"], entry["outcome"]["code"])

    def on_history_selected(self, item, _previous=None):
        if item is None:
            return
        self.show_entry(self.entries.get(item.data(Qt.UserRole)))

    def show_entry(self, entry):
        if entry is None:
            return
        self.current_id = entry["id"]
        r, o = entry["result"], entry["outcome"]
        jpeg = scanner.captures.get(entry["id"])
        if jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(jpeg)
            self.crop.setPixmap(pixmap.scaled(self.crop.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.crop.setText("(photo no longer kept)")
        f = self.fields
        f["outcome"].setText("%s — %s" % (o["code"], o["title"]) + ("" if r.get("ok") else "  (%s)" % r.get("error")))
        f["timings"].setText("total %.1f s" % (r.get("timings", {}).get("total_ms", 0) / 1000))

    # ------------------------------------------------------------------ controls

    def capture(self):
        scanner.submit("capture", source="manual")

    def relearn(self):
        scanner.submit("relearn")

    def pause(self, paused):
        scanner.submit("pause", paused)
        self.pause_button.setText("Resume  [P]" if paused else "Pause  [P]")

    def auto_scan(self, on):
        scanner.submit("auto_scan", on)

    def set_review_mode(self, on):
        scanner.submit("approval", on)

    def test_image(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Run test images through the scanner", "",
                                                "Images (*.jpg *.jpeg *.png *.bmp)")
        for path in paths:
            try:
                with open(path, "rb") as f:
                    jpeg = normalize_image(f.read(), cfg["detection"]["max_image_side"])
            except (OSError, ValueError) as e:
                self.append_log("Test image skipped (%s): %s" % (type(e).__name__, path.rsplit("/", 1)[-1]), "warn")
                continue
            scanner.submit("image", jpeg, "test")

    def closeEvent(self, event):
        self.video_timer.stop()
        self.status_timer.stop()
        self.pump.stop()
        engine.stop()
        if getattr(self, "api_server", None) is not None:
            self.api_server.should_exit = True
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    window = Console()
    scanner.approval_ui = True          # this console shows the approval window for traced notes
    engine.start("desktop console")
    if cfg.get("desktop", {}).get("api", True):
        try:
            from .main import run_api_in_thread
            window.api_server = run_api_in_thread()
        except Exception as e:
            hub.log("API not started: %s" % e, "warn")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
