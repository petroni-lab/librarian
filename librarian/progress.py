"""A tiny animated terminal spinner for showing pipeline progress.

Standard library only — no tqdm, no dependencies. Renders a braille spinner plus
a status message and an elapsed-seconds counter on a single line of stderr, the
way modern CLIs (uv, Claude Code, ...) do. Feed it the current stage via
``update()``; it is thread-safe, so worker threads can report progress too.

On a non-TTY (output piped/redirected) it silently does nothing, so logs and the
JSON result on stdout stay clean.
"""

import itertools
import shutil
import sys
import threading
import time

# Braille dots — the smooth, single-cell spinner used by most modern CLIs.
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_CLEAR_LINE = "\r\033[2K"  # carriage return + ANSI "erase entire line"


class Spinner:
    """Animated single-line progress spinner (context manager)."""

    def __init__(self, stream=None, interval: float = 0.08):
        self._stream = stream or sys.stderr
        self._interval = interval
        self._enabled = self._stream.isatty()
        self._text = "Working…"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._start = 0.0

    def update(self, text: str) -> None:
        """Set the status message shown next to the spinner (thread-safe)."""
        with self._lock:
            self._text = text

    def __enter__(self) -> "Spinner":
        if self._enabled:
            self._start = time.monotonic()
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self._enabled:
            self._stream.write(_CLEAR_LINE)
            self._stream.flush()

    def _animate(self) -> None:
        for frame in itertools.cycle(_FRAMES):
            if self._stop.is_set():
                return
            with self._lock:
                text = self._text
            elapsed = int(time.monotonic() - self._start)
            line = f"{frame} {text} ({elapsed}s)"
            width = shutil.get_terminal_size((80, 20)).columns
            self._stream.write(_CLEAR_LINE + line[: width - 1])
            self._stream.flush()
            time.sleep(self._interval)
