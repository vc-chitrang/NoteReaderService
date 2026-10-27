r"""
Scores the note reader on saved photos, through exactly the same code the service uses.

    .venv\Scripts\python -m tests.benchmark                       (profile from config.json)
    .venv\Scripts\python -m tests.benchmark --profile accurate --samples tests\indic_samples

Each samples folder has the photos and a truth.json: {"file.jpg": {"name": "...", "message": "..."}}
(use "" for a missing name or message). The message is compared before the English tidy pass.
"""
import argparse
import difflib
import json
import os
import re
import sys

from app import config
from app.detector import normalize_image
from app.events import EventHub
from app.reader import NoteReader

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(text):
    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()


def similarity(a, b):
    a, b = norm(a), norm(b)
    return 1.0 if not a and not b else difflib.SequenceMatcher(None, a, b).ratio()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--samples", default=os.path.join(HERE, "samples"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config.load()
    reader = NoteReader(cfg["reader"], EventHub(os.path.join(config.ROOT, "logs")))
    if args.profile:
        reader.profile = args.profile
    if not reader.prepare():
        raise SystemExit("Reader not ready: " + reader.status)

    truth = json.load(open(os.path.join(args.samples, "truth.json"), encoding="utf-8"))
    name_exact = name_like = message_like = seconds = 0.0
    for file, want in truth.items():
        with open(os.path.join(args.samples, file), "rb") as f:
            jpeg = normalize_image(f.read(), cfg["reader"]["max_image_side"])
        got = reader.read(jpeg)
        ns = similarity(got["name"], want["name"])
        ms = similarity(got["message_before_tidy"], want["message"])
        name_exact += norm(got["name"]) == norm(want["name"])
        name_like += ns
        message_like += ms
        seconds += got["timings"].get("total_ms", 0) / 1000
        print("%s %-24s %5.1fs  name %-14s want %-14s  msg %3.0f%%  %s" % (
            "ok" if ns == 1.0 and ms > 0.9 else "--", file, got["timings"].get("total_ms", 0) / 1000,
            repr(got["name"]), repr(want["name"]), ms * 100, got["error"]))

    n = len(truth)
    print("\n%s (%s): name exact %.0f%%, name like %.0f%%, message like %.0f%%, avg %.1f s" % (
        reader.model, reader.profile, name_exact / n * 100, name_like / n * 100, message_like / n * 100, seconds / n))


if __name__ == "__main__":
    main()
