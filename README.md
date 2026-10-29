# Good Deed Tree - Note Scanner

Scans handwritten sticky notes from a camera and sends the note's pen strokes to the wall app (Unity),
which draws them in neon on the leaf. **No AI is used**: the strokes come from plain image processing
(black and white, clean-up, thinning, tracing). Everything runs offline on this PC.

## Use
1. One time: double-click `install.bat` (creates `.venv` and installs the Python packages).
2. Every time: double-click `start.bat`. It opens the **Note Scanner** desktop console.
   Close the console window to stop it.
3. Keep the desk empty at start (the empty desk is learned after ~2.5 s), or press **R** later.
4. Start the wall app (Unity). Notes go to `http://127.0.0.1:8080/api/v1/notes` on this PC.

## How a note is handled
- A note placed on the desk and held still is traced and sent to the wall automatically.
- A different note placed over the previous one is detected and sent too.
- The same note again (even moved a little) is recognised from its ink and not sent twice
  (`handwriting.duplicate_seconds`, default 90 s).
- Tick **Review before sending** in the console to see original / black and white / cleaned / strokes
  and approve each note first (**A** approve, **X** reject in the review window).

## Console (desktop, PySide6)
- Live camera with the detected note box, the scan state and "Wall would show" feedback.
- Controls: **C** capture now, **R** relearn empty desk, **P** pause, Auto scan, Review before sending,
  **Test image...** (runs saved photos through the same path).
- Last note card, history of the last 50 notes (memory only) and the live log.

## Settings (`config.json`, restart to apply)
- `camera.name_contains` (default `Iriun`), or `camera.index`; `camera.rotate` 0/90/180/270.
- `detection.*`: threshold, note size, how long a note must be still.
- `handwriting.wall_url` (must be on this PC), `require_approval`, `duplicate_seconds`,
  `watch_for_new_note`, `new_note_change`.

## Local API (127.0.0.1:8765, optional)
`GET /health`, `GET /api/status`, `POST /api/capture|relearn|pause|auto-scan|review`, and
`POST /v1/tasks/sticky-note` (image bytes in, `{aspect, strokes}` out; sends nothing).

## Privacy
Listens on 127.0.0.1 only. Photos are kept in memory (last 20, for the console) and never written to
disk. `logs/service.log` has timings and counts only, never names or messages.
