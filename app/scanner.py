"""
The scanning loop (one worker thread). Watches the camera, waits for a note to be placed and held
still, crops it, traces its pen strokes (plain image processing, no AI) and sends them to the wall, and
publishes every step to the console. Manual captures, uploaded test images and API requests are queued
to the same worker, so only one note is processed at a time.
"""
import collections
import json
import queue
import threading
import time
import urllib.request
import uuid

import cv2

from . import strokes
from .detector import Detector

TICK = 0.1


class Job:
    def __init__(self, kind, payload=None, source=""):
        self.kind = kind
        self.payload = payload
        self.source = source
        self.done = threading.Event()
        self.result = None


class Scanner:
    def __init__(self, cfg, camera, hub):
        self.cfg = cfg
        self.det_cfg = cfg["detection"]
        self.camera = camera
        self.hub = hub
        self.detector = Detector(self.det_cfg)

        self.state = "starting"
        self.message = None              # what the wall would show: {kind, title, sub}
        self.paused = False
        self.auto_scan = bool(self.det_cfg.get("auto_scan", True))
        self.captures = collections.OrderedDict()    # id -> JPEG bytes, memory only
        self.previews = collections.OrderedDict()    # id -> {stage: PNG bytes} for the approval window, memory only
        self.approval_ui = False                     # the desktop console sets this: it will show the approval window
        self._decisions = {}                         # id -> [threading.Event, approved]
        self._sent = collections.deque(maxlen=10)    # (fingerprint, time) of notes already sent: no duplicates
        self._eval_sig = None                        # the picture when the last note was read (to spot a new note)
        self._new_still = 0.0
        self._last_eval = 0.0
        self.history = collections.deque(maxlen=50)
        self.reads = 0
        self.total_read_ms = 0

        self._jobs = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="scanner", daemon=True)
        self._next_id = 1
        self._state_since = time.time()
        self._still_for = 0.0
        self._empty_for = 0.0
        self._light_change_for = 0.0
        self._cut_off_shown = False
        self._take_prompt_shown = False
        self._camera_since = None
        self._last_frame_id = -1
        self._result_until = 0.0
        self._after_result = "idle"

    # ------------------------------------------------------------------ public (any thread)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def submit(self, kind, payload=None, source=""):
        job = Job(kind, payload, source)
        self._jobs.put(job)
        return job

    @property
    def queue_length(self):
        return self._jobs.qsize()

    def status(self):
        return {
            "state": self.state,
            "message": self.message,
            "paused": self.paused,
            "auto_scan": self.auto_scan,
            "fraction": round(self.detector.fraction, 4),
            "motion": round(self.detector.motion, 4),
            "still_for": round(self._still_for, 1),
            "bbox": self.detector.bbox,
            "has_reference": self.detector.has_reference,
            "reads": self.reads,
            "avg_read_ms": round(self.total_read_ms / self.reads) if self.reads else 0,
            "queue": self.queue_length,
        }

    # ------------------------------------------------------------------ worker

    def _run(self):
        self._enter("starting")
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=TICK)
            except queue.Empty:
                job = None
            try:
                if job is not None:
                    self._handle(job)
                else:
                    self._tick()
            except Exception as e:      # keep the loop alive whatever happens
                self.hub.log("Scanner error: %s: %s" % (type(e).__name__, e), "error")
                if job is not None:
                    job.result = {"error": "internal error"}
                    job.done.set()

    def _handle(self, job):
        if job.kind == "image":
            job.result = self._read(job.payload, job.source)
            self._after_result = "idle"
        elif job.kind == "capture":
            frame, _ = self.camera.latest()
            if frame is None:
                job.result = {"error": "camera not ready"}
                self.hub.log("Manual capture ignored: camera not ready.", "warn")
            else:
                if self.detector.has_reference:
                    self.detector.update(Detector.sample(frame))
                self.hub.log("Manual capture (%s)." % ("note box" if self.detector.bbox else "whole view"))
                jpeg = self.detector.crop_jpeg(frame, self.det_cfg["max_image_side"])
                self._eval_sig = self._signature(frame)
                job.result = self._read(jpeg, "manual")
                self._after_result = "waiting_for_removal"
        elif job.kind == "relearn":
            frame, _ = self.camera.latest()
            if frame is not None:
                self.detector.take_reference(Detector.sample(frame))
                self.hub.log("Desk reference relearned.")
                self._enter("idle")
        elif job.kind == "pause":
            self.paused = bool(job.payload)
            self.hub.log("Scanning paused." if self.paused else "Scanning resumed.")
            self._enter("paused" if self.paused else "idle")
        elif job.kind == "approval":
            self.cfg["handwriting"]["require_approval"] = bool(job.payload)
            self.hub.log("Review before sending %s." % ("on" if job.payload else "off (notes are sent automatically)"))
        elif job.kind == "auto_scan":
            self.auto_scan = bool(job.payload)
            self.hub.log("Auto scan %s." % ("on" if self.auto_scan else "off (manual capture only)"))
        job.done.set()

    def _tick(self):
        frame, frame_id = self.camera.latest()
        if frame is None:
            self._camera_since = None
            if self.state != "no_camera" and not self.paused:
                self._enter("no_camera")
            return
        if self._camera_since is None:
            self._camera_since = time.time()
        if not self.detector.has_reference:
            # Let auto-exposure settle, then learn what the empty desk looks like.
            if time.time() - self._camera_since >= self.det_cfg["warmup_seconds"]:
                self.detector.take_reference(Detector.sample(frame))
                self.hub.log("Desk reference learned (keep the desk empty at start, or press R).")
                self._enter("idle")
            return
        if self.state == "no_camera":
            self._enter("idle")

        if self.state == "result":
            if time.time() >= self._result_until:
                self._enter(self._after_result)
            return
        if self.paused or frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id

        small = Detector.sample(frame)
        fraction, motion = self.detector.update(small)
        c = self.det_cfg
        note_like = c["min_note_fraction"] <= fraction <= c["max_note_fraction"]

        if self.state == "idle":
            if note_like and self.auto_scan:
                self._enter("stabilizing", "info", "I can see your note", "Hold it still for a moment")
            elif fraction > c["max_note_fraction"]:
                # The whole picture changed (lights, camera bumped): relearn the desk once it settles.
                self._light_change_for = self._light_change_for + TICK if motion < c["still_threshold"] else 0.0
                if self._light_change_for >= 2.0:
                    self.detector.take_reference(small)
                    self._light_change_for = 0.0
                    self.hub.log("Desk reference relearned after a lighting change.")
            elif not note_like:
                self._light_change_for = 0.0
                self.detector.track_lighting(small)

        elif self.state == "stabilizing":
            if not note_like:
                self._enter("idle")
                return
            self._still_for = self._still_for + TICK if motion < c["still_threshold"] else 0.0
            cut_off = self.detector.touches_edge()
            if cut_off and self._still_for >= c["still_seconds"]:
                # A note running off the picture would be read with missing letters.
                if not self._cut_off_shown:
                    self._cut_off_shown = True
                    self._show("warning", "Please place your note fully in view",
                               "Move it a little away from the camera so nothing is cut off")
                    self.hub.log("Note touches the edge of the picture; not reading.", "warn")
                self._still_for = 0.0
                return
            if not cut_off and self._cut_off_shown:
                self._cut_off_shown = False
                self._show("info", "I can see your note", "Hold it still for a moment")
            if self._still_for >= c["still_seconds"]:
                jpeg = self.detector.crop_jpeg(frame, self.det_cfg["max_image_side"])
                self._eval_sig = self._signature(frame)
                self._read(jpeg, "camera")
                self._after_result = "waiting_for_removal"

        elif self.state == "waiting_for_removal":
            if fraction < c["min_note_fraction"] * 0.75:
                self._empty_for += TICK
                if self._empty_for >= 1.0:
                    self._enter("idle")
            else:
                self._empty_for = 0.0
                self._watch_for_new_note(frame, small, note_like, motion)
                if not self._take_prompt_shown and time.time() - self._state_since > c["take_note_prompt_seconds"]:
                    self._take_prompt_shown = True
                    self._show("info", "Please take your note", "so the next person can add theirs")

    def _watch_for_new_note(self, frame, small, note_like, motion):
        """A different note placed over (or instead of) the last one is read and sent too, without waiting for
        the desk to be emptied. The same note again is recognised by its ink and not sent twice."""
        hw = self.cfg["handwriting"]
        if not (hw.get("watch_for_new_note", True) and self.auto_scan and note_like):
            self._new_still = 0.0
            return
        c = self.det_cfg
        if self._eval_sig is None or time.time() - self._last_eval < 2.0:
            return
        # Compare at a size where handwriting is visible (the 64x36 desk grid is too coarse to see a changed word).
        signature = self._signature(frame)
        changed = float((abs(signature - self._eval_sig) > 45.0).mean()) > hw.get("new_note_change", 0.012)
        if not changed or motion >= c["still_threshold"] or self.detector.touches_edge():
            self._new_still = 0.0
            return
        self._new_still += TICK
        if self._new_still >= c["still_seconds"]:
            self._new_still = 0.0
            self._last_eval = time.time()
            self._eval_sig = signature
            self.hub.log("A different note is on the desk; checking it.")
            jpeg = self.detector.crop_jpeg(frame, self.det_cfg["max_image_side"])
            self._read(jpeg, "camera-new")
            self._after_result = "waiting_for_removal"

    @staticmethod
    def _signature(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA), (5, 5), 0).astype("float32")

    def _is_duplicate(self, fingerprint):
        window = self.cfg["handwriting"].get("duplicate_seconds", 90)
        now = time.time()
        return any(now - t <= window and strokes.same_note(fp, fingerprint) for fp, t in self._sent)

    # ------------------------------------------------------------------ reading

    def _read(self, jpeg, source):
        return self._handwriting(jpeg, source)

    def decide(self, capture_id, approved):
        """Called by the console when Approve / Reject is pressed for a traced note."""
        pending = self._decisions.get(capture_id)
        if pending:
            pending[1] = bool(approved)
            pending[0].set()

    @staticmethod
    def _png(image, width):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (width, max(1, int(h * width / float(w)))), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".png", resized)[1].tobytes()

    def _stage_pngs(self, traced):
        stages = traced["stages"]
        return {"raw": self._png(stages["raw"], 700), "bw": self._png(stages["bw"], 700),
                "clean": self._png(stages["clean"], 700), "final": self._png(strokes.render_overlay(traced, stages["clean"], 700), 700)}

    def _await_approval(self, capture_id, traced):
        """Debug gate: show the stages in the console and wait for Approve / Reject. True = send to the wall."""
        hw = self.cfg["handwriting"]
        if not (hw.get("require_approval") and self.approval_ui):
            return True
        self.previews[capture_id] = self._stage_pngs(traced)
        while len(self.previews) > self.cfg["privacy"]["keep_last_captures"]:
            self.previews.popitem(last=False)
        pending = [threading.Event(), False]
        self._decisions[capture_id] = pending
        self._enter("reading", "info", "Waiting for you to check the note...", "Approve it in the console")
        self.hub.emit("approval", text="Note #%d traced; waiting for approval." % capture_id, data={"id": capture_id})
        deadline = time.time() + hw.get("approval_timeout_seconds", 180)
        while not pending[0].wait(0.5):
            if self._stop.is_set() or time.time() > deadline:
                break
        self._decisions.pop(capture_id, None)
        return pending[1]

    def _handwriting(self, jpeg, source):
        """Handwriting mode: trace the note's pen strokes and send them to the wall. Nothing is read."""
        capture_id = self._store_capture(jpeg)
        self._enter("reading", "info", "Getting your note ready...", "Hold on a moment")
        self.hub.emit("capture", data={"capture_id": capture_id, "source": source, "bytes": len(jpeg)})
        started = time.time()
        decision = {"code": "", "kind": "warning", "title": "", "sub": "Then place it under the camera again",
                    "name": "", "message": "", "language": "handwriting"}
        stroke_count = 0
        try:
            traced = strokes.extract(jpeg, stages=True)
            stroke_count = len(traced["strokes"])
            if stroke_count == 0:
                decision.update(code="unreadable", title="We couldn't see any writing",
                                sub="Please write a little bigger and darker, then place it again")
            elif self._is_duplicate(strokes.fingerprint(traced["stages"]["clean"])):
                decision.update(code="duplicate", kind="info", title="This note was already sent",
                                sub="Place a new note when you are ready")
            elif not self._await_approval(capture_id, traced):
                decision.update(code="not_approved", title="Note not sent",
                                sub="Please place it again if you want to try once more")
            else:
                reply = self._send_to_wall(traced)
                if reply.get("status") == "accepted":
                    fp = strokes.fingerprint(traced["stages"]["clean"])
                    if fp is not None:
                        self._sent.append((fp, time.time()))
                    decision.update(code="accepted", kind="success", title="Thank you!",
                                    sub="Your leaf is on its way to the tree")
                else:
                    decision.update(code="wall_refused", title="The tree couldn't take your note",
                                    sub="Please try again in a moment")
        except ValueError:
            decision.update(code="unreadable", title="We couldn't see your note",
                            sub="Please place it flat under the camera")
        except OSError as e:     # the wall app is not running or not reachable
            decision.update(code="wall_offline", title="The tree is resting",
                            sub="Please try again in a moment")
            self.hub.log("Wall not reachable: %s" % type(e).__name__, "warn")

        total_ms = int((time.time() - started) * 1000)
        result = {"ok": decision["code"] == "accepted", "name": "", "message": "", "language": "handwriting",
                  "error": "" if decision["code"] == "accepted" else decision["code"], "timings": {"total_ms": total_ms}}
        self.reads += 1
        self.total_read_ms += total_ms
        entry = {"id": capture_id, "t": time.time(), "source": source, "result": result, "outcome": decision}
        self.history.appendleft(entry)
        # Log line on disk: timing and counts only, never the visitor's writing.
        self.hub.emit("result", level="info" if result["ok"] else "warn", data=entry, text=(
            "Note #%d (%s): %d strokes in %.1f s -> %s" % (capture_id, source, stroke_count, total_ms / 1000.0, decision["code"])))
        self._enter("result", decision["kind"], decision["title"], decision["sub"])
        self._result_until = time.time() + self.det_cfg["result_seconds"]
        return entry

    def _send_to_wall(self, traced):
        hw = self.cfg["handwriting"]
        body = json.dumps({"requestId": uuid.uuid4().hex, "aspect": traced["aspect"],
                           "strokes": traced["strokes"]}).encode("utf-8")
        request = urllib.request.Request(hw["wall_url"], data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=hw["timeout_seconds"]) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def _store_capture(self, jpeg):
        capture_id = self._next_id
        self._next_id += 1
        self.captures[capture_id] = jpeg
        while len(self.captures) > self.cfg["privacy"]["keep_last_captures"]:
            self.captures.popitem(last=False)
        return capture_id

    # ------------------------------------------------------------------ state

    def _enter(self, state, kind=None, title=None, sub=None):
        self.state = state
        self._state_since = time.time()
        self._still_for = 0.0
        self._empty_for = 0.0
        self._take_prompt_shown = False
        self._cut_off_shown = False
        self.message = {"kind": kind, "title": title, "sub": sub} if title else None
        self.hub.emit("state", data={"state": state, "message": self.message})

    def _show(self, kind, title, sub):
        self.message = {"kind": kind, "title": title, "sub": sub}
        self.hub.emit("state", data={"state": self.state, "message": self.message})
