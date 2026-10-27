"""
Note detection, ported from the wall app's NoteScanner. The live picture is shrunk to a 64x36 grid
and compared with a reference picture of the empty desk, so any note colour works and no markers
are needed. The biggest changed area is the note; its box is used to crop the full-size photo.
"""
import cv2
import numpy as np

GW, GH = 64, 36


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reference = None
        self.previous = None
        self.bbox = None           # (x0, y0, x1, y1) in grid cells, inclusive
        self.fraction = 0.0
        self.motion = 0.0

    @property
    def has_reference(self):
        return self.reference is not None

    @staticmethod
    def sample(frame):
        return cv2.resize(frame, (GW, GH), interpolation=cv2.INTER_AREA).astype(np.float32)

    def take_reference(self, small):
        self.reference = small.copy()
        self.previous = None

    def update(self, small):
        """Compares a sampled frame with the desk and the previous frame. Returns (fraction, motion)."""
        diff = np.abs(small - self.reference).max(axis=2)
        mask = (diff > self.cfg["pixel_threshold"] * 255.0).astype(np.uint8)
        self.fraction = float(mask.mean())

        # The note is the biggest connected changed area; stray noisy cells are ignored.
        self.bbox = None
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, w, h, _ = stats[best]
            self.bbox = (int(x), int(y), int(x + w - 1), int(y + h - 1))

        self.motion = 0.0 if self.previous is None else float(np.abs(small - self.previous).mean() / 255.0)
        self.previous = small
        return self.fraction, self.motion

    def track_lighting(self, small, rate=0.03):
        """Slowly follows daylight and exposure drift while the desk is empty."""
        self.reference += (small - self.reference) * rate

    def touches_edge(self):
        if self.bbox is None:
            return False
        x0, y0, x1, y1 = self.bbox
        return x0 <= 0 or y0 <= 0 or x1 >= GW - 1 or y1 >= GH - 1

    def crop_jpeg(self, frame, max_side=1280, quality=90, use_box=True):
        """Crops the note (with a margin) from the full-size frame and encodes a JPEG."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.bbox if (use_box and self.bbox) else (0, 0, GW - 1, GH - 1)
        pad_x = (x1 - x0 + 1) * 0.08 + 1
        pad_y = (y1 - y0 + 1) * 0.08 + 1
        px0 = int(max(0.0, (x0 - pad_x) / GW) * w)
        px1 = int(min(1.0, (x1 + 1 + pad_x) / GW) * w)
        py0 = int(max(0.0, (y0 - pad_y) / GH) * h)
        py1 = int(min(1.0, (y1 + 1 + pad_y) / GH) * h)
        crop = frame[py0:max(py1, py0 + 16), px0:max(px1, px0 + 16)]
        return encode_jpeg(crop, max_side, quality)


def encode_jpeg(image, max_side=1280, quality=90):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode JPEG")
    return buf.tobytes()


def normalize_image(data, max_side=1280):
    """Decodes any uploaded image, then re-encodes it the way camera captures are. Raises ValueError if unreadable."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image")
    return encode_jpeg(image, max_side)
