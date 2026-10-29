"""
Note scanner HTTP API (local only, 127.0.0.1). Runs in the background of the desktop console.

    GET  /health                    camera and scanner state
    GET  /api/status                scanner state, camera
    POST /api/capture               scan the note on the desk now
    POST /api/relearn               relearn the empty desk
    POST /api/pause                 {"paused": true|false}
    POST /api/auto-scan             {"enabled": true|false}
    POST /api/review                {"enabled": true|false}   ask before sending
    POST /v1/tasks/sticky-note      body = image bytes (JPEG/PNG) -> {aspect, strokes}   (traces only, sends nothing)

No AI is used: strokes come from plain image processing. Photos stay in memory; nothing personal is
written to disk.
"""
import contextlib
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from . import __version__, engine, strokes
from .detector import normalize_image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024

cfg, hub, camera, scanner = engine.cfg, engine.hub, engine.camera, engine.scanner


@contextlib.asynccontextmanager
async def lifespan(_app):
    # The desktop console starts the engine itself; start() is a no-op then.
    engine.start("API on http://%s:%d" % (cfg["host"], cfg["port"]))
    yield
    engine.stop()


app = FastAPI(title="Good Deed Tree note scanner", version=__version__, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if camera.ok else "degraded", "version": __version__, "camera": camera.ok}


@app.get("/api/status")
def status():
    return {"version": __version__, "scanner": scanner.status(),
            "camera": {"ok": camera.ok, "device": camera.device_name, "size": camera.size, "fps": round(camera.fps, 1)}}


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
    scanner.submit("pause", bool((await request.json()).get("paused")))
    return {"queued": True}


@app.post("/api/auto-scan")
async def auto_scan(request: Request):
    scanner.submit("auto_scan", bool((await request.json()).get("enabled")))
    return {"queued": True}


@app.post("/api/review")
async def review(request: Request):
    scanner.submit("approval", bool((await request.json()).get("enabled")))
    return {"queued": True}


@app.post("/v1/tasks/sticky-note")
async def trace_sticky_note(request: Request):
    """Traces one note photo into pen strokes and returns them. Does not send anything to the wall."""
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body: send the image bytes")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 8 MB")
    try:
        traced = strokes.extract(normalize_image(data, cfg["detection"]["max_image_side"]))
    except ValueError:
        raise HTTPException(400, "not a readable image")
    return {"aspect": traced["aspect"], "strokes": traced["strokes"]}


def run_api_in_thread():
    """Serves the HTTP API from a background thread (used by the desktop console). Returns the server."""
    server = uvicorn.Server(uvicorn.Config(app, host=cfg["host"], port=cfg["port"], log_level="warning",
                                           timeout_graceful_shutdown=2, lifespan="off"))
    threading.Thread(target=server.run, name="api", daemon=True).start()
    hub.log("API listening on http://%s:%d (local only)." % (cfg["host"], cfg["port"]))
    return server


def main():
    uvicorn.run(app, host=cfg["host"], port=cfg["port"], log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
