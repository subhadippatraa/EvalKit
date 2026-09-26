"""Exclusive execution of a run (audit P1-2): at most one executor per run at a time.

Two executors on one run would both call the target and the judge for the same pending cases:
double spend, duplicate-result errors, and a run finished under the other's feet. The design's
answer to that (leases, multi-process workers) is P2; this is the small piece P1 needs -- one
executor per run, on one host.

The lock is an OS advisory lock (`flock`) on `<db>.locks/<run_id>.lock`. It is held by an open file
descriptor, so it disappears the instant its holder dies: a `kill -9` never leaves a stale lock and
there is nothing to time out or clean up. `flock` conflicts between two descriptors even inside one
process, which is what makes two `execute()` calls in the same interpreter exclude each other too.
An in-memory database has no directory, so a process-wide registry covers it. Platforms without
`fcntl` (Windows) fall back to the registry: exclusion inside one process only (documented).

Local filesystems only, like the SQLite database itself (network filesystems lie about flock).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from evalkit.errors import RunBusyError

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]

_held: set[str] = set()  # process-wide registry: (db path or ":memory:" identity) + run id
_registry_lock = threading.Lock()


class RunLock:
    """Context manager: `with RunLock(lock_dir, run_id, scope):`. Raises `RunBusyError`."""

    def __init__(self, directory: Path | None, run_id: str, scope: str):
        self.directory, self.run_id = directory, run_id
        self._key = f"{scope}\0{run_id}"
        self._fd: int | None = None
        self._registered = False

    def __enter__(self) -> RunLock:
        with _registry_lock:
            if self._key in _held:
                raise self._busy()
            _held.add(self._key)
            self._registered = True
        try:
            if self.directory is not None and fcntl is not None:
                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(self.directory / f"{self.run_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    raise self._busy() from None
                self._fd = fd
        except BaseException:
            self._release_registry()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)  # closing the descriptor releases the flock
            finally:
                self._fd = None
        self._release_registry()

    def _release_registry(self) -> None:
        if self._registered:
            with _registry_lock:
                _held.discard(self._key)
            self._registered = False

    def _busy(self) -> RunBusyError:
        return RunBusyError(
            f"run {self.run_id} is already being executed by another executor (this process or "
            "another one on this host); wait for it to finish, or stop it, before executing again"
        )
