"""
Reads a sticky-note photo with the local vision model (Ollama on 127.0.0.1). Returns the visitor's
name, the good deed, the language and whether the note was readable. Prompts are the ones tuned
in the wall app (NoteReader.cs), so results match what was benchmarked.
"""
import base64
import json
import time
import urllib.error
import urllib.request

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "message": {"type": "string"},
        "language": {"type": "string", "enum": ["en", "hi", "gu", "other"]},
        "readable": {"type": "boolean"},
    },
    "required": ["name", "message", "language", "readable"],
}

SYSTEM_PROMPT = (
    "You read handwritten sticky notes for an art installation called the Good Deed Tree. "
    "Each note should contain a person's first NAME and a short GOOD DEED they promise to do. "
    "Transcribe exactly what is written. Fix only obvious spelling mistakes. "
    "Never translate, never add words, never invent a name or a message. "
    "Write the name and the message in the same script the note is written in: Latin letters for English, "
    "Devanagari for Hindi, Gujarati script for Gujarati. Never transliterate a name into Latin letters. "
    "The name may follow a label such as 'Name:' or appear as a signature such as '- Anna'. "
    "If there is no name, return an empty string for name. "
    "If there is no good deed, return an empty string for message. "
    "Return the message as a short phrase exactly as the visitor wrote it, for example 'Help a stranger'. "
    "Set readable to false only if there is no legible text at all; a note that has only a name, or only a good deed, is still readable. "
    "Set language to 'en' for English, 'hi' for Hindi (Devanagari), 'gu' for Gujarati, otherwise 'other'. "
    "Examples: a note that says only 'Name: Rahul' gives name 'Rahul', empty message, readable true. "
    "A note that says only 'Help a stranger' gives an empty name, message 'Help a stranger', readable true. "
    "Reply only with JSON.")

USER_PROMPT = "Read this sticky note and return the JSON."

TIDY_SCHEMA = {"type": "object", "properties": {"deed": {"type": "string"}}, "required": ["deed"]}

TIDY_PROMPT = (
    "You tidy a short good deed written by a visitor so it reads naturally after the words 'chose to'. "
    "Rewrite it as an action phrase in simple present base form. Fix spelling, grammar and capitalisation; "
    "start with a lower-case verb; add small missing words like 'a' or 'the'. "
    "Keep every action and the full meaning; never add new actions. Do not translate. "
    "Examples: 'Helped Stranger and feed the Street Dogs' -> 'help a stranger and feed the street dogs'; "
    "'I will call my mother every day.' -> 'call my mother every day'; "
    "'feeding birds' -> 'feed the birds'; 'Planted tree' -> 'plant a tree'; "
    "'Save water' -> 'save water'. Reply only with JSON.")


def clean(text):
    if not text:
        return ""
    text = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    return text.strip(" \"'-:.")


def fix_swapped(name, message):
    """
    Smaller models sometimes put the deed in the name field when a note has no name. A first name is
    1-3 words; anything longer is the deed, so it moves to the message and the name stays empty.
    """
    if len(name.split()) > 3:
        return "", (name + " " + message if message else name)
    if message and name.lower() == message.lower():
        return "", message
    return name, message


def detect_language(text, model_guess):
    """'hi' if Devanagari letters are present, 'gu' for Gujarati, 'en' for Latin text, else the model's guess."""
    latin = False
    for c in text:
        if "ऀ" <= c <= "ॿ":
            return "hi"
        if "઀" <= c <= "૿":
            return "gu"
        if ("a" <= c <= "z") or ("A" <= c <= "Z"):
            latin = True
    return "en" if latin else (model_guess or "other")


def _short_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        return ("HTTP %d %s" % (exc.code, detail)).strip()[:160]
    if isinstance(exc, urllib.error.URLError):
        return "AI service not reachable (%s)" % exc.reason
    return type(exc).__name__ + (": " + str(exc)[:120] if str(exc) else "")


class NoteReader:
    def __init__(self, cfg, hub):
        self.cfg = cfg
        self.hub = hub
        self.endpoint = cfg["ollama"].rstrip("/")
        self.profile = cfg["profile"]
        self.ready = False
        self.status = "not started"

    @property
    def model(self):
        return self.cfg["profiles"][self.profile]["model"]

    @property
    def options(self):
        # Every call must use the same options, or Ollama reloads the model between calls.
        return {**self.cfg["options"], **self.cfg["profiles"][self.profile].get("options", {})}

    # ------------------------------------------------------------------ plumbing

    def _post(self, path, body, timeout):
        request = urllib.request.Request(self.endpoint + path, json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def installed_models(self):
        with urllib.request.urlopen(self.endpoint + "/api/tags", timeout=4) as response:
            return [m["name"] for m in json.loads(response.read().decode("utf-8")).get("models", [])]

    # ------------------------------------------------------------------ lifecycle

    def prepare(self):
        """Checks Ollama is up and the model installed, then loads it into GPU memory."""
        self.ready = False
        try:
            models = self.installed_models()
        except Exception as e:
            self.status = _short_error(e)
            self.hub.log("Note reader not ready: " + self.status, "warn")
            return False
        if self.model not in models:
            self.status = "model '%s' is not installed" % self.model
            self.hub.log("Note reader not ready: " + self.status, "warn")
            return False
        self.status = "loading " + self.model
        started = time.perf_counter()
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": -1, "prompt": "",
                                         "options": self.options}, self.cfg.get("load_timeout_seconds", 600))
        except Exception as e:
            self.status = "load failed: " + _short_error(e)
            self.hub.log("Model %s failed to load: %s" % (self.model, _short_error(e)), "error")
            return False
        self.ready = True
        self.status = "ready"
        self.hub.log("Model %s loaded in %.1f s." % (self.model, time.perf_counter() - started))
        return True

    def unload(self, model):
        try:
            self._post("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""}, 30)
            self.hub.log("Model %s unloaded from GPU memory." % model)
        except Exception as e:
            self.hub.log("Could not unload %s: %s" % (model, _short_error(e)), "warn")

    def profile_for(self, model):
        """Profile key for an installed model name, adding a plain profile for models not in config.json."""
        for key, profile in self.cfg["profiles"].items():
            if profile["model"] == model:
                return key
        self.cfg["profiles"][model] = {"model": model}
        return model

    def set_profile(self, profile):
        if profile not in self.cfg["profiles"]:
            raise ValueError("unknown profile " + profile)
        old = self.model
        self.profile = profile
        if old != self.model:
            self.unload(old)     # two models in 12 GB next to the wall scene would not fit
        return self.prepare()

    # ------------------------------------------------------------------ reading

    def read(self, jpeg):
        started = time.perf_counter()
        result = {"ok": False, "error": "", "readable": False, "name": "", "message": "", "language": "other",
                  "message_before_tidy": "", "raw": None, "model": self.model, "profile": self.profile,
                  "timings": {}}
        body = {
            "model": self.model, "stream": False, "keep_alive": -1, "format": SCHEMA, "options": self.options,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT, "images": [base64.b64encode(jpeg).decode("ascii")]},
            ],
        }
        try:
            response = self._post("/api/chat", body, self.cfg["timeout_seconds"])
            content = (response.get("message") or {}).get("content", "")
            if not content:
                raise ValueError(response.get("error") or "empty reply")
            note = json.loads(content)
        except Exception as e:
            result["error"] = _short_error(e)
            result["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000)
            return result

        result["timings"].update(_ollama_timings(response))
        result["raw"] = note
        name, message = fix_swapped(clean(note.get("name")), clean(note.get("message")))
        result.update(ok=True, readable=bool(note.get("readable")), name=name, message=message,
                      message_before_tidy=message,
                      # The model's own language guess is unreliable; the script of the letters is not.
                      language=detect_language(name + " " + message, note.get("language")))

        # Second, quick text-only pass: make the deed read well after "chose to".
        if self.cfg.get("tidy_english", True) and message and result["language"] == "en":
            tidy_started = time.perf_counter()
            tidied = self._tidy(message)
            result["timings"]["tidy_ms"] = round((time.perf_counter() - tidy_started) * 1000)
            if tidied:
                result["message"] = tidied

        result["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000)
        return result

    def _tidy(self, deed):
        """Rewrites a deed as a clean action phrase. Returns None on any problem (the original is kept)."""
        body = {
            "model": self.model, "stream": False, "keep_alive": -1, "format": TIDY_SCHEMA, "options": self.options,
            "messages": [{"role": "system", "content": TIDY_PROMPT}, {"role": "user", "content": deed}],
        }
        try:
            response = self._post("/api/chat", body, 30)
            text = clean(json.loads(response["message"]["content"]).get("deed"))
        except Exception:
            return None
        # Guard against the model running off: accept only a sensible length.
        return text if 0 < len(text) <= max(60, len(deed) * 2) else None


def _ollama_timings(response):
    """Ollama reports durations in nanoseconds. A non-zero load time means the model was (re)loaded."""
    ms = lambda key: round(response.get(key, 0) / 1e6)
    return {"model_ms": ms("total_duration"), "load_ms": ms("load_duration"),
            "prompt_ms": ms("prompt_eval_duration"), "generate_ms": ms("eval_duration")}
