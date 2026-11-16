#!/usr/bin/env python3
"""Keep macOS awake while the Layer Atlas Docker services are running."""

import fcntl
import os
import signal
import subprocess
import threading

from manage import COMPOSE, ROOT


def services_running():
    try:
        result = subprocess.run(COMPOSE + ["ps", "--status", "running", "--services"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and {"atlas", "gateway"}.issubset(result.stdout.splitlines())


def release(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def watch():
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    inhibitor = None
    try:
        while not stopping.is_set():
            if services_running():
                if inhibitor is None or inhibitor.poll() is not None:
                    inhibitor = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())])
                    print("Layer Atlas running: preventing idle sleep; display sleep remains allowed.", flush=True)
            elif inhibitor is not None:
                release(inhibitor)
                inhibitor = None
                print("Layer Atlas stopped: released sleep prevention.", flush=True)
            stopping.wait(30)
    finally:
        release(inhibitor)


if __name__ == "__main__":
    os.umask(0o077)
    (ROOT / "data").mkdir(exist_ok=True)
    with (ROOT / "data" / "awake.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A Layer Atlas keep-awake watcher is already running")
        watch()
