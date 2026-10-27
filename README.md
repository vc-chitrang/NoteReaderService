# Good Deed Tree - Note Reader

Reads handwritten sticky notes (`Name: ____` / `Message: ____`) from a camera with a local AI model.
Runs fully offline on this PC: the camera, note detection, AI reading and checks all happen here.
The wall app (Unity) will only receive the finished result. For now results stay in this console
(test mode) until they are confirmed.

## Use
1. One time: double-click `install.bat` (creates `.venv`, installs packages, checks the models).
2. Every time: double-click `start.bat`. It starts Ollama if needed and opens the **Note Reader**
   desktop console. Close the console window to stop it.
3. Keep the desk empty at start (the empty desk is learned after ~2.5 s), or press **R** later.

This service owns the camera and all note reading. The wall app (Unity) has no camera or AI code;
it will only receive finished results. For now results stay in this console (play-test mode).

## Console (desktop, PySide6)
- Live camera with the detected note box, a scan line while reading, and the scan state.
- "Wall would show": the exact message the wall would display (or nothing).
- Controls: **C** capture now, **R** relearn empty desk, **P** pause, auto scan on/off,
  **Test image...** (runs saved photos through the same path), AI model dropdown (all installed
  Ollama models; switching unloads the old one from GPU memory first).
- Read card: cropped photo, the wall sentence, name, deed (after the tidy and as written),
  language, outcome, timings, model and the raw model answer.
- Play test: **A** approve / **X** reject each read. Nothing is sent to the wall yet.
- History of the last 50 reads and the live log.

Browser console (old, optional): `.venv\Scripts\python -m app.main` -> http://127.0.0.1:8765/

## Settings (`config.json`, restart to apply)
- `camera.name_contains` (default `Iriun`), or `camera.index`; `camera.rotate` 0/90/180/270.
- `detection.*`: same tuning as the wall app (threshold, note size, still time).
- `reader.profiles`: `fast` = gemma3:4b, `accurate` = gemma3:12b. Switch in the console;
  the old model is unloaded from GPU memory first.
- `reader.english_only`, `limits.name_max` / `message_max`.

## API (for the wall app)
- `GET /health` -> `{status, camera, model, model_loaded}`
- `POST /v1/tasks/sticky-note` with the image bytes as the body -> `{ok, id, result, outcome}`.
  `outcome.code`: `accepted`, `missing_name`, `missing_message`, `unreadable`,
  `unsupported_language`, `reader_not_ready`. `outcome.title`/`sub` are the wall messages.
  One read at a time; 429 `busy` if more than 3 are waiting.

## Accuracy check
`.venv\Scripts\python -m tests.benchmark [--profile accurate] [--samples tests\indic_samples]`

## Privacy
Listens on 127.0.0.1 only. Photos are kept in memory (last 20, for the console) and never
written to disk. `logs/service.log` has timings and yes/no flags only, never names or messages.
