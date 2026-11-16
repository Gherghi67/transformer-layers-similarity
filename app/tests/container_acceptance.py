"""Exercise only the disposable layer-atlas-verify Compose project on port 18786."""

from contextlib import closing
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manage", ROOT / "manage.py")
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)
COMPOSE = [manage.DOCKER, "compose", "-p", "layer-atlas-verify", "-f", str(ROOT / "compose.yaml")]
ENV = {**os.environ, "LAYER_ATLAS_PORT": "18786"}


def command(*args, **kwargs):
    return subprocess.run(COMPOSE + list(args), env=ENV, check=True, timeout=90, **kwargs)


def request(method, path, payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", 18786, timeout=10)
    connection.request(method, path, json.dumps(payload) if payload is not None else None,
                       {"Content-Type": "application/json", **(headers or {})})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


identifier = command("ps", "-q", "atlas", capture_output=True, text=True).stdout.strip()
details = json.loads(subprocess.check_output([manage.DOCKER, "inspect", identifier]))[0]
host = details["HostConfig"]
assert details["Config"]["User"] == "10001:10001"
assert host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"]
assert "no-new-privileges:true" in host["SecurityOpt"]
assert host["Memory"] == host["MemorySwap"] == 512 * 1024**2
assert host["NanoCpus"] == 10**9 and host["PidsLimit"] == 64
assert not host["PortBindings"]
assert len(details["Mounts"]) == 1 and details["Mounts"][0]["Type"] == "volume"
network = json.loads(subprocess.check_output([manage.DOCKER, "network", "inspect", "layer-atlas-verify_isolated"]))[0]
assert network["Internal"]
gateway_id = command("ps", "-q", "gateway", capture_output=True, text=True).stdout.strip()
gateway = json.loads(subprocess.check_output([manage.DOCKER, "inspect", gateway_id]))[0]
assert gateway["HostConfig"]["PortBindings"]["8786/tcp"] == [{"HostIp": "127.0.0.1", "HostPort": "18786"}]
assert gateway["Config"]["User"] == "101:101" and not gateway["Mounts"]
assert gateway["HostConfig"]["ReadonlyRootfs"] and gateway["HostConfig"]["CapDrop"] == ["ALL"]
isolation = """
import os, pathlib, socket
assert os.getuid() == 10001
assert 'CapEff:\\t0000000000000000' in pathlib.Path('/proc/self/status').read_text()
assert 'NoNewPrivs:\\t1' in pathlib.Path('/proc/self/status').read_text()
for name in ['/app/should-not-write', '/etc/should-not-write']:
    try:
        pathlib.Path(name).write_text('test')
    except OSError:
        pass
    else:
        raise AssertionError(name + ' is writable')
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not pathlib.Path('/Users').exists()
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2).close()
except OSError:
    pass
else:
    raise AssertionError('Unexpected outbound internet access')
print('PASS: non-root, read-only, capabilities removed, no host mounts, outbound internet blocked')
"""
command("exec", "-T", "atlas", "python", "-c", isolation)
assert request("GET", "/api/health")[0] == 200
assert request("GET", "/")[0] == 200
assert request("GET", "/api/health", headers={"Host": "bad.example"})[0] == 403
assert request("GET", "/data/layer-atlas.sqlite3")[0] == 404
example = json.loads((ROOT / "examples/example.json").read_text())
assert request("POST", "/api/runs", example, {"Origin": "https://bad.example"})[0] == 403
status, body = request("POST", "/api/runs", example)
assert status == 201, body
run = json.loads(body)
path = "/api/runs/" + run["id"]
assert request("PATCH", path, {"notes": "Docker restore fixture", "revision": 1})[0] == 200
assert request("PATCH", "/api/tasks/1", {"done": True})[0] == 200
assert json.loads(request("GET", path + "/export.json")[1])["matrix"] == example["matrix"]
with tempfile.TemporaryDirectory() as directory:
    manage.COMPOSE = COMPOSE
    manage.BACKUPS = Path(directory)
    manage.backup()
    snapshot = next(manage.BACKUPS.glob("*.sqlite3"))
    initial_mtime = snapshot.stat().st_mtime_ns
    manage.backup()
    assert snapshot.stat().st_mtime_ns == initial_mtime, "Daily backup should not overwrite itself"
    with closing(sqlite3.connect(snapshot)) as db:
        assert db.execute("SELECT done FROM tasks WHERE id=1").fetchone() == (1,)
        assert "Docker restore fixture" in db.execute("SELECT payload FROM runs WHERE id=?", (run["id"],)).fetchone()[0]
    assert request("DELETE", path, {})[0] == 200
    assert request("PATCH", "/api/tasks/1", {"done": False})[0] == 200
    command("stop", "atlas")
    restore = """
from pathlib import Path
import sys
for suffix in ['-wal', '-shm']:
    Path('/data/layer-atlas.sqlite3' + suffix).unlink(missing_ok=True)
Path('/data/layer-atlas.sqlite3').write_bytes(sys.stdin.buffer.read())
"""
    with snapshot.open("rb") as source:
        command("run", "--rm", "-T", "--no-deps", "atlas", "python", "-c", restore, stdin=source)
    command("up", "-d", "--no-build", "--wait", "--wait-timeout", "45")
    assert json.loads(request("GET", path)[1])["notes"] == "Docker restore fixture"
    assert json.loads(request("GET", "/api/tasks")[1])[0]["done"] is True
print("PASS: imports, notes, exports, host/origin checks, daily backup and recovery after deletion and restart")
