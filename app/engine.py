"""
The note scanner engine: config, camera and scanner, wired once and shared by every
front end (the desktop console and the HTTP API). Start it once; starting again is a no-op.
"""
import os
import threading
import time

import cv2

from . import __version__, config
from .camera import Camera
from .detector import GW, GH
from .events import EventHub
from .scanner import Scanner

cfg = config.load()
hub = EventHub(os.path.join(config.ROOT, "logs"))
camera = Camera(cfg["camera"], hub)
scanner = Scanner(cfg, camera, hub)

_started = False
_lock = threading.Lock()


def start(front_end):
    global _started
    with _lock:
        if _started:
            return
        _started = True
    hub.log("Note scanner %s starting (%s)." % (__version__, front_end))
    camera.start()
    scanner.start()


def stop():
    global _started
    with _lock:
        if not _started:
            return
        _started = False
    hub.log("Shutting down.")
    scanner.stop()
    camera.stop()


# ---------------------------------------------------------------------- camera overlay

STATE_COLOURS = {"idle": (160, 160, 160), "stabilizing": (0, 200, 255), "reading": (255, 160, 0),
                 "result": (80, 220, 80), "waiting_for_removal": (200, 120, 255), "paused": (90, 90, 90),
                 "no_camera": (60, 60, 220)}


def overlay(frame, width=960):
    """Scaled copy of the frame with the detected note box, the scan line and the scan state drawn on."""
    h, w = frame.shape[:2]
    scale = float(width) / w
    view = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    vh, vw = view.shape[:2]
    colour = STATE_COLOURS.get(scanner.state, (200, 200, 200))
    bbox = scanner.detector.bbox
    if bbox and scanner.state != "paused":
        x0, y0, x1, y1 = bbox
        p0 = (int(x0 / GW * vw), int(y0 / GH * vh))
        p1 = (int((x1 + 1) / GW * vw), int((y1 + 1) / GH * vh))
        cv2.rectangle(view, p0, p1, colour, 2)
    if scanner.state == "reading":
        # Scan bar sweeping down, like the wall's feedback panel.
        y = int((time.time() * 0.8 % 1.0) * vh)
        cv2.line(view, (0, y), (vw, y), (255, 160, 0), 2)
    label = scanner.state.replace("_", " ")
    cv2.rectangle(view, (0, 0), (vw, 28), (20, 20, 20), -1)
    cv2.putText(view, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    cv2.putText(view, "change %.1f%%  motion %.3f" % (scanner.detector.fraction * 100, scanner.detector.motion),
                (vw - 280, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return view
