"""
Note reader HTTP API + browser console.

    .venv\Scripts\python -m app.main          browser console + API (headless)
    .venv\Scripts\python -m app.desktop       desktop console; runs this API in the background

Console:  http://127.0.0.1:8765/         live camera, scan state, results, logs
API:      GET  /health
          POST /v1/tasks/sticky-note     body = image bytes (JPEG/PNG) -> read result + outcome
Local only (127.0.0.1). Photos stay in memory; nothing personal is written to disk.
"""
import asyncio
import contextlib
import json
import os
import threading
import time
import webbrowser

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from . import __version__, engine
from .camera import list_devices
from .detector import normalize_image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

cfg, hub, camera, reader, scanner = engine.cfg, engine.hub, engine.camera, engine.reader, engine.scanner


@contextlib.asynccontextmanager
async def lifespan(_app):
    # The desktop console starts the engine itself; start() is a no-op then.
    engine.start("API on http://%s:%d" % (cfg["host"], cfg["port"]))
    yield
    engine.stop()


app = FastAPI(title="Good Deed Tree note reader", version=__version__, lifespan=lifespan)


# ---------------------------------------------------------------------- console

@app.get("/")
def console():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return {
        "version": __version__,
        "scanner": scanner.status(),
        "camera": {"ok": camera.ok, "device": camera.device_name, "size": camera.size, "fps": round(camera.fps, 1)},
        "reader": {"ready": reader.ready, "status": reader.status, "profile": reader.profile, "model": reader.model,
                   "profiles": {k: v["model"] for k, v in cfg["reader"]["profiles"].items()}},
        "unity_output": False,   # results are only shown here until the developer confirms them
    }


@app.get("/api/history")
def history():
    return list(scanner.history)


@app.get("/api/devices")
def devices():
    return {"devices": list_devices(), "in_use": camera.device_name}


@app.get("/api/captures/{capture_id}.jpg")
def capture_image(capture_id: int):
    jpeg = scanner.captures.get(capture_id)
    if jpeg is None:
        raise HTTPException(404, "capture no longer kept")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/capture")
def capture():
    scanner.submit("capture")
    return {"queued": True}


@app.post("/api/relearn")
def relearn():
    scanner.submit("relearn")
    return {"queued": True}


@app.post("/api/pause")
async def pause(request: Request):
    body = await request.json()
    scanner.submit("pause", bool(body.get("paused")))
    return {"queued": True}


@app.post("/api/auto-scan")
async def auto_scan(request: Request):
    body = await request.json()
    scanner.submit("auto_scan", bool(body.get("enabled")))
    return {"queued": True}


@app.post("/api/profile")
async def set_profile(request: Request):
    profile = (await request.json()).get("profile")
    if profile not in cfg["reader"]["profiles"]:
        raise HTTPException(400, "unknown profile")
    scanner.submit("profile", profile)
    return {"queued": True}


@app.post("/api/test-image")
async def test_image(request: Request):
    """Developer console upload: runs a saved photo through the same path as a camera capture."""
    jpeg = await _read_image(request)
    scanner.submit("image", jpeg, "upload")
    return {"queued": True}


@app.get("/api/events")
async def events(request: Request):
    """Server-sent events: logs, state changes and results, as they happen."""
    async def stream():
        last = 0
        while not await request.is_disconnected():
            items = await asyncio.to_thread(hub.since, last, 10.0)
            if not items:
                yield ": keep-alive\n\n"
                continue
            last = items[-1]["seq"]
            for item in items:
                yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@app.get("/video")
def video():
    """Live camera as MJPEG, with the detected note box and scan state drawn on top."""
    def frames():
        last_id = -1
        while True:
            frame, frame_id = camera.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.05)
                continue
            last_id = frame_id
            view = engine.overlay(frame)
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1 / 15)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------- service API (for Unity later)

@app.get("/health")
def health():
    return {"status": "ok" if reader.ready else "degraded", "version": __version__, "camera": camera.ok,
            "model": reader.model, "model_loaded": reader.ready, "reader_status": reader.status}


@app.post("/v1/tasks/sticky-note")
async def read_sticky_note(request: Request):
    """Reads one note photo and returns {result, outcome}. One read at a time; others wait in line."""
    if scanner.queue_length >= 3:
        return JSONResponse({"ok": False, "error": {"code": "busy", "message": "reader is busy"}}, status_code=429)
    jpeg = await _read_image(request)
    job = scanner.submit("image", jpeg, "api")
    finished = await asyncio.to_thread(job.done.wait, cfg["reader"]["timeout_seconds"] + 30)
    if not finished:
        return JSONResponse({"ok": False, "error": {"code": "timeout", "message": "read timed out"}}, status_code=504)
    entry = job.result or {}
    if "result" not in entry:
        return JSONResponse({"ok": False, "error": {"code": "internal", "message": entry.get("error", "")}}, status_code=500)
    return {"ok": entry["result"]["ok"], "id": entry["id"], "result": entry["result"], "outcome": entry["outcome"]}


async def _read_image(request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body: send the image bytes")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 8 MB")
    try:
        return normalize_image(data, cfg["reader"]["max_image_side"])
    except ValueError:
        raise HTTPException(400, "not a readable image")


def run_api_in_thread():
    """Serves the HTTP API from a background thread (used by the desktop console). Returns the server."""
    server = uvicorn.Server(uvicorn.Config(app, host=cfg["host"], port=cfg["port"], log_level="warning",
                                           timeout_graceful_shutdown=2, lifespan="off"))
    threading.Thread(target=server.run, name="api", daemon=True).start()
    hub.log("API listening on http://%s:%d (local only)." % (cfg["host"], cfg["port"]))
    return server


def main():
    if cfg.get("open_browser", True):
        url = "http://%s:%d/" % (cfg["host"], cfg["port"])
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=cfg["host"], port=cfg["port"], log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
