#!/usr/bin/env python3
"""Host-side service maintenance and daily SQLite backups. No host mounts needed."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
BACKUPS = ROOT / "data" / "backups"
DOCKER = shutil.which("docker") or "/usr/local/bin/docker"
COMPOSE = [DOCKER, "compose", "-f", str(ROOT / "compose.yaml")]

SNAPSHOT = """
from contextlib import closing
import os, sqlite3, sys, tempfile
fd, name = tempfile.mkstemp(prefix='.snapshot-', suffix='.sqlite3', dir='/data')
os.close(fd)
try:
    with closing(sqlite3.connect('file:/data/layer-atlas.sqlite3?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(name)) as target:
            source.backup(target)
    with open(name, 'rb') as snapshot:
        while block := snapshot.read(1024 * 1024):
            sys.stdout.buffer.write(block)
finally:
    os.unlink(name)
"""


def verify_backup(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("SQLite integrity check failed")
        db.execute("SELECT id, payload, revision FROM runs LIMIT 1").fetchall()
        db.execute("SELECT id, done FROM tasks LIMIT 1").fetchall()


def backup(force=False):
    BACKUPS.mkdir(parents=True, exist_ok=True, mode=0o700)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    destination = BACKUPS / f"layer-atlas-{today}.sqlite3"
    if destination.exists() and not force:
        verify_backup(destination)
        return
    fd, temporary = tempfile.mkstemp(prefix=".backup-", suffix=".sqlite3", dir=BACKUPS)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            subprocess.run(COMPOSE + ["exec", "-T", "atlas", "python", "-c", SNAPSHOT],
                           stdout=output, check=True, timeout=180)
            output.flush()
            os.fsync(output.fileno())
        verify_backup(temporary)
        os.replace(temporary, destination)
        for old in sorted(BACKUPS.glob("layer-atlas-????-??-??.sqlite3"), reverse=True)[30:]:
            old.unlink()
        print(f"Verified backup: {destination}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def maintain():
    ready = subprocess.run([DOCKER, "info", "--format", "{{.ServerVersion}}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    if ready.returncode:
        subprocess.run([DOCKER, "desktop", "start", "--timeout", "60"], check=True, timeout=75)
    result = subprocess.run(COMPOSE + ["up", "-d", "--no-build", "--wait", "--wait-timeout", "45"],
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    backup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["maintain", "backup"])
    args = parser.parse_args()
    os.umask(0o077)
    (ROOT / "data").mkdir(exist_ok=True)
    with (ROOT / "data" / "maintenance.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if args.action == "maintain":
            maintain()
        else:
            backup(force=True)


if __name__ == "__main__":
    main()
