"""
Camera capture on its own thread. Always holds the newest frame; reconnects by itself when the
camera (for example the Iriun phone webcam) drops out.
"""
import threading
import time

import cv2

_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def list_devices():
    """DirectShow camera names, in the same order OpenCV numbers them."""
    try:
        import comtypes
        comtypes.CoInitialize()
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


class Camera:
    def __init__(self, cfg, hub):
        self.cfg = cfg
        self.hub = hub
        self.device_name = None
        self.size = (0, 0)
        self.fps = 0.0
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    @property
    def ok(self):
        return self._frame is not None and self.device_name is not None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def latest(self):
        """(frame, frame_id). The frame is shared: copy it before drawing on it."""
        with self._lock:
            return self._frame, self._frame_id

    def _resolve(self):
        names = list_devices()
        index = int(self.cfg.get("index", -1))
        if index >= 0:
            return index, names[index] if index < len(names) else "camera %d" % index
        want = (self.cfg.get("name_contains") or "").lower()
        for i, name in enumerate(names):
            if want and want in name.lower():
                return i, name
        self.hub.log("Camera '%s' not found. Cameras seen: %s" % (self.cfg.get("name_contains"), names or "none"), "warn")
        return None, None

    def _open(self):
        index, name = self._resolve()
        if index is None:
            return None
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("width", 1920)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("height", 1080)))
        cap.set(cv2.CAP_PROP_FPS, int(self.cfg.get("fps", 30)))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok or frame is None:
            cap.release()
            self.hub.log("Camera '%s' opened but gave no picture (is another app using it?)." % name, "warn")
            return None
        self.device_name = name
        self.size = (frame.shape[1], frame.shape[0])
        self.hub.log("Camera ready: %s %dx%d" % (name, frame.shape[1], frame.shape[0]))
        return cap

    def _run(self):
        rotate = _ROTATIONS.get(int(self.cfg.get("rotate", 0)))
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._stop.wait(3.0)
                continue
            failures = 0
            window_start, window_frames = time.time(), 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 40:
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                if rotate is not None:
                    frame = cv2.rotate(frame, rotate)
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                window_frames += 1
                if time.time() - window_start >= 2.0:
                    self.fps = window_frames / (time.time() - window_start)
                    window_start, window_frames = time.time(), 0
            cap.release()
            with self._lock:
                self._frame = None
            self.device_name = None
            self.fps = 0.0
            if not self._stop.is_set():
                self.hub.log("Camera lost; reconnecting...", "warn")
