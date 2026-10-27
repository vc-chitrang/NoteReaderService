"""
Event hub: every log line, state change and read result goes through here. The console streams
these to the browser; log lines also go to logs/service.log.

Privacy rule: `text` is written to disk, so it must never contain a visitor's name or message.
Personal data goes only in `data`, which stays in memory for the developer console.
"""
import collections
import logging
import logging.handlers
import os
import threading
import time

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}


class EventHub:
    def __init__(self, log_dir, size=600):
        self._cond = threading.Condition()
        self._items = collections.deque(maxlen=size)
        self._seq = 0

        os.makedirs(log_dir, exist_ok=True)
        self._logger = logging.getLogger("notereader")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s")
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, "service.log"), maxBytes=2_000_000, backupCount=5, encoding="utf-8")
            file_handler.setFormatter(fmt)
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console)

    def emit(self, kind, text=None, data=None, level="info"):
        with self._cond:
            self._seq += 1
            self._items.append({"seq": self._seq, "t": time.time(), "kind": kind, "level": level,
                                "text": text, "data": data})
            self._cond.notify_all()
        if text:
            self._logger.log(_LEVELS.get(level, logging.INFO), "%-6s %s", kind, text)

    def log(self, text, level="info"):
        self.emit("log", text=text, level=level)

    def since(self, seq, timeout=15.0):
        """Events newer than `seq`; waits up to `timeout` seconds for one to arrive."""
        with self._cond:
            if not self._items or self._items[-1]["seq"] <= seq:
                self._cond.wait(timeout)
            return [item for item in self._items if item["seq"] > seq]
