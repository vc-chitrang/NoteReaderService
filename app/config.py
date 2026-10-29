"""Loads config.json from the service folder. Missing keys fall back to the defaults below."""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": True,
    "camera": {"name_contains": "Iriun", "index": -1, "width": 1920, "height": 1080, "fps": 30, "rotate": 0},
    "detection": {
        "auto_scan": True,
        "pixel_threshold": 0.12,
        "min_note_fraction": 0.02,
        "max_note_fraction": 0.7,
        "still_seconds": 1.2,
        "still_threshold": 0.012,
        "warmup_seconds": 2.5,
        "result_seconds": 4.5,
        "take_note_prompt_seconds": 14,
        "max_image_side": 1280,
    },
    "privacy": {"keep_last_captures": 20},
    # Desktop console: also serve the HTTP API in the background (for the wall app later).
    "desktop": {"api": True},
    # Handwriting mode: no reading. The note's own pen strokes are traced and sent to the wall,
    # which draws them in neon on the leaf. The wall URL must be on this PC.
    "handwriting": {"wall_url": "http://127.0.0.1:8080/api/v1/notes", "timeout_seconds": 5,
                    # Debug: in the desktop console, show original / black-and-white / cleaned / neon strokes
                    # and send to the wall only after Approve. Set false to send straight away.
                    "require_approval": False, "approval_timeout_seconds": 180,
                    # Automatic mode: a note is traced and sent as soon as it is placed and held still; a different
                    # note placed over it is sent too. A note already sent in the last duplicate_seconds is not sent again.
                    "duplicate_seconds": 90, "watch_for_new_note": True, "new_note_change": 0.012},
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and key != "profiles":
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load(path=None):
    path = path or os.path.join(ROOT, "config.json")
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _merge(cfg, json.load(f))
    # Never listen on the network: this service handles visitors' handwriting.
    if cfg["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("config.json 'host' must be 127.0.0.1 (local only)")
    if not cfg["handwriting"]["wall_url"].startswith(("http://127.0.0.1", "http://localhost")):
        raise ValueError("config.json handwriting.wall_url must point at this PC (127.0.0.1)")
    return cfg
