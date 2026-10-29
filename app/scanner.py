"""
The scanning loop (one worker thread). Watches the camera, waits for a note to be placed and held
still, crops it, reads it with the AI, decides the outcome, and publishes every step to the console.
Manual captures, uploaded test images and API requests are queued to the same worker, so only one
note is ever read at a time (the GPU is shared with the wall scene).
"""
import collections
import json
import queue
import threading
import time
import urllib.request
import uuid

import cv2

from . import outcome as outcomes
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
    def __init__(self, cfg, camera, reader, hub):
        self.cfg = cfg
        self.det_cfg = cfg["detection"]
        self.camera = camera
        self.reader = reader
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
        self._preparing = threading.Lock()
        self._next_prepare = 0.0

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

    def _prepare_reader_async(self):
        """Loads the model in the background; retried every 10 s while Ollama is down."""
        self._next_prepare = time.time() + 10.0
        if self._preparing.acquire(blocking=False):
            def work():
                try:
                    self.reader.prepare()
                finally:
                    self._preparing.release()
            threading.Thread(target=work, name="reader-prepare", daemon=True).start()

    def _run(self):
        handwriting = self.cfg["handwriting"]["enabled"]
        if not handwriting:      # handwriting mode never loads the AI model (it shares the GPU with the wall)
            self._prepare_reader_async()
        self._enter("starting")
        while not self._stop.is_set():
            if not handwriting and not self.reader.ready and time.time() >= self._next_prepare:
                self._prepare_reader_async()
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
                jpeg = self.detector.crop_jpeg(frame, self.cfg["reader"]["max_image_side"])
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
        elif job.kind == "auto_scan":
            self.auto_scan = bool(job.payload)
            self.hub.log("Auto scan %s." % ("on" if self.auto_scan else "off (manual capture only)"))
        elif job.kind == "model":
            job.kind, job.payload = "profile", self.reader.profile_for(job.payload)
            self._handle(job)
            return
        elif job.kind == "profile":
            self._enter("loading_model", "info", "Loading the AI model...", self.reader.cfg["profiles"][job.payload]["model"])
            with self._preparing:
                ok = self.reader.set_profile(job.payload)
            job.result = {"ok": ok, "profile": self.reader.profile, "model": self.reader.model, "status": self.reader.status}
            self._enter("idle")
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
                jpeg = self.detector.crop_jpeg(frame, self.cfg["reader"]["max_image_side"])
                self._read(jpeg, "camera")
                self._after_result = "waiting_for_removal"

        elif self.state == "waiting_for_removal":
            if fraction < c["min_note_fraction"] * 0.75:
                self._empty_for += TICK
                if self._empty_for >= 1.0:
                    self._enter("idle")
            else:
                self._empty_for = 0.0
                if not self._take_prompt_shown and time.time() - self._state_since > c["take_note_prompt_seconds"]:
                    self._take_prompt_shown = True
                    self._show("info", "Please take your note", "so the next person can add theirs")

    # ------------------------------------------------------------------ reading

    def _read(self, jpeg, source):
        if self.cfg["handwriting"]["enabled"]:
            return self._handwriting(jpeg, source)
        capture_id = self._store_capture(jpeg)
        self._enter("reading", "info", "Reading your note...", "This takes a few seconds")
        self.hub.emit("capture", data={"capture_id": capture_id, "source": source, "bytes": len(jpeg)})

        if not self.reader.ready:
            with self._preparing:     # Ollama may have been started after this service
                if not self.reader.ready:
                    self.reader.prepare()
        result = self.reader.read(jpeg)
        decision = outcomes.evaluate(result, self.cfg["limits"], self.cfg["reader"].get("english_only", False))

        self.reads += 1
        self.total_read_ms += result["timings"].get("total_ms", 0)
        entry = {"id": capture_id, "t": time.time(), "source": source, "result": result, "outcome": decision}
        self.history.appendleft(entry)

        # Log line on disk: timing and flags only, never the visitor's words.
        t = result["timings"]
        self.hub.emit("result", level="info" if result["ok"] else "warn", data=entry, text=(
            "Read #%d (%s) in %.1f s [model %.1f s, tidy %.1f s, load %.1f s]: ok=%s lang=%s name=%s deed=%s -> %s%s" % (
                capture_id, source, t.get("total_ms", 0) / 1000, t.get("model_ms", 0) / 1000,
                t.get("tidy_ms", 0) / 1000, t.get("load_ms", 0) / 1000, result["ok"], result["language"],
                "yes" if result["name"] else "no", "yes" if result["message"] else "no", decision["code"],
                "" if result["ok"] else " error=" + result["error"])))

        self._enter("result", decision["kind"], decision["title"], decision["sub"])
        self._result_until = time.time() + self.det_cfg["result_seconds"]
        return entry

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
            elif not self._await_approval(capture_id, traced):
                decision.update(code="not_approved", title="Note not sent",
                                sub="Please place it again if you want to try once more")
            else:
                reply = self._send_to_wall(traced)
                if reply.get("status") == "accepted":
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
