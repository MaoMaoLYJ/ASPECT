"""Bounded cleanup of an owned process tree, including detached GPU engines."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.children = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.capture()
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def capture(self):
        with self.lock:
            roots = [(self.process.pid, None), *self.children.items()]
            covered = set()
            for pid, created in roots:
                if pid in covered:
                    continue
                try:
                    parent = psutil.Process(pid)
                    if created is not None and parent.create_time() != created:
                        continue
                    for child in parent.children(recursive=True):
                        self.children[child.pid] = child.create_time()
                        covered.add(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    def _track(self):
        while not self.done.wait(.1):
            self.capture()

    def stop(self, grace=10):
        self.capture()
        self.done.set()
        self.thread.join(timeout=2)
        owned = []
        with self.lock:
            for pid, created in self.children.items():
                try:
                    child = psutil.Process(pid)
                    if child.create_time() == created:
                        owned.append(child)
                except psutil.NoSuchProcess:
                    pass
        # Callers launch the root in a private session. Group signals also
        # cover children forked in the interval between tracker samples.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for child in owned:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=grace)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, survivors = psutil.wait_procs(alive, timeout=grace)
        live = []
        for child in survivors:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    live.append(child.pid)
            except psutil.NoSuchProcess:
                pass
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Owned process did not exit after SIGKILL: {self.process.pid}") from exc
        if live:
            raise RuntimeError(f"Owned descendants survived SIGKILL: {live}")


class ProgressWatchdog:
    def __init__(self, path, startup_timeout, idle_timeout, now=None):
        self.path = Path(path)
        self.startup_timeout, self.idle_timeout = startup_timeout, idle_timeout
        self.last_change = time.monotonic() if now is None else now
        self.signature = None
        self.started = False

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        try:
            stat = self.path.stat()
            # Network filesystems may coarsen mtimes. Small progress records
            # can change without changing their length or reported timestamp.
            content = self.path.read_bytes() if stat.st_size <= 4096 else None
            signature = (stat.st_mtime_ns, stat.st_size, content)
            if stat.st_size and signature != self.signature:
                self.signature, self.last_change, self.started = signature, now, True
        except FileNotFoundError:
            pass
        limit = self.idle_timeout if self.started else self.startup_timeout
        if now - self.last_change > limit:
            phase = "progress" if self.started else "startup"
            raise TimeoutError(f"No real {phase} for {limit}s: {self.path}")
