#!/usr/bin/env python3
"""Layer Atlas: a local result store and buildless research workbench."""

import argparse
from contextlib import closing, contextmanager
import csv
from datetime import datetime, timezone
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit
import uuid

LOGGER = logging.getLogger("layer_atlas")

ROOT = Path(__file__).resolve().parent
MAX_BODY = 25 * 1024 * 1024
METRICS = {"cosine", "centered_cosine", "linear_cka", "js_divergence"}
TEXT_FIELDS = {"name": 200, "model": 200, "task": 100, "pooling": 100}
OPTIONAL_TEXT = {"split": 100, "notes": 20000, "provenance": 4000}
TASKS = [
    ("Load Pythia-410M in a Colab notebook and collect every layer's hidden states", "2026-11-21"),
    ("Build loaders for GSM8K, ARC-Challenge, HellaSwag, WinoGrande, LAMBADA and GLUE with a fixed sample of 300–500", "2026-11-22"),
    ("Check 12 GB GPU fit for Qwen2.5-1.5B, Gemma 3 1B, Qwen3.5-0.8B and ModernBERT; Falcon-H1 optional", "2026-11-22"),
    ("Reproduce Fig. 14 with the Pythia-410M cosine map on LAMBADA", "2026-11-24"),
    ("Inspect token lengths, truncation and GLUE class balance", "2026-11-24"),
    ("Compute the LAMBADA map with linear CKA and mean-centered cosine", "2026-11-26"),
    ("Measure model accuracy on each task and mark performance above chance", "2026-11-27"),
    ("Build layer maps for five decoder tasks across Pythia 70M–1.4B and reasoning minus semantic difference maps", "2026-11-29"),
    ("Build Pythia-410M head maps using attention hooks and Jensen–Shannon divergence", "2026-11-29"),
    ("Build layer maps for Qwen2.5-1.5B, Gemma 3 1B and Qwen3.5-0.8B on the same tasks", "2026-11-30"),
    ("Map BERT-base on GLUE, fine-tune on SST-2 and RTE, then repeat the maps", "2026-11-30"),
    ("Write progress report 1 with figures by hand and submit before 10:00 on 03 Dec", "2026-12-02"),
]


def validate_run(value):
    if not isinstance(value, dict):
        raise ValueError("The import must be a JSON object.")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be 1.")
    result = {"schema_version": 1}
    for key, limit in TEXT_FIELDS.items():
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > limit:
            raise ValueError(f"{key} must be nonempty text, up to {limit} characters.")
        result[key] = text.strip()
    for key, limit in OPTIONAL_TEXT.items():
        text = value.get(key, "")
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError(f"{key} must be text, up to {limit} characters.")
        result[key] = text
    if value.get("kind") not in ("layer", "head"):
        raise ValueError("kind must be layer or head.")
    if not isinstance(value.get("metric"), str) or value["metric"] not in METRICS:
        raise ValueError("metric must be cosine, centered_cosine, linear_cka or js_divergence.")
    result.update(kind=value["kind"], metric=value["metric"])
    for key in ("sample_count", "max_tokens", "seed"):
        number = value.get(key)
        if number is None and key != "sample_count":
            result[key] = None
            continue
        minimum = 0 if key == "seed" else 1
        if type(number) is not int or not minimum <= number <= 2**53 - 1:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
        result[key] = number
    for key in ("accuracy", "chance_accuracy"):
        number = value.get(key)
        if number is not None and (type(number) not in (int, float) or not 0 <= number <= 1 or not math.isfinite(number)):
            raise ValueError(f"{key} must be a fraction between 0 and 1, or null.")
        result[key] = number
    labels = value.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 512:
        raise ValueError("labels must contain between 2 and 512 entries.")
    if any(not isinstance(label, str) or not label.strip() or len(label) > 100 for label in labels):
        raise ValueError("Each label must be nonempty text, up to 100 characters.")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique.")
    matrix = value.get("matrix")
    size = len(labels)
    if not isinstance(matrix, list) or len(matrix) != size or any(not isinstance(row, list) or len(row) != size for row in matrix):
        raise ValueError("matrix must be square, with one row and column per label.")
    lower = -1 if result["metric"] in ("cosine", "centered_cosine") else 0
    for row in matrix:
        for number in row:
            if type(number) not in (int, float):
                raise ValueError("matrix entries must be finite numbers.")
            if not lower - 1e-6 <= number <= 1 + 1e-6:
                raise ValueError(f"{result['metric']} entries must be between {lower} and 1.")
            if not math.isfinite(number):
                raise ValueError("matrix entries must be finite numbers.")
    for i in range(size):
        for j in range(i):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                raise ValueError("matrix must be symmetric within 0.000001.")
    result.update(labels=labels, matrix=matrix)
    return result


def warnings_for(run):
    diagonal = 0 if run["metric"] == "js_divergence" else 1
    if any(abs(row[i] - diagonal) > 1e-6 for i, row in enumerate(run["matrix"])):
        return [f"Some diagonal entries differ from the expected {diagonal}. Check degenerate representations and your metric implementation."]
    return []


def initialize(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL, due TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0)")
        db.execute("""CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            actor TEXT NOT NULL, identity_source TEXT NOT NULL, action TEXT NOT NULL,
            target TEXT NOT NULL, details TEXT NOT NULL, request_id TEXT NOT NULL)
        """)
        db.executemany("INSERT OR IGNORE INTO tasks(id,title,due) VALUES(?,?,?)", [(i, title, due) for i, (title, due) in enumerate(TASKS, 1)])


def unpack(row):
    run = json.loads(row["payload"])
    run.update(id=row["id"], created_at=row["created_at"], revision=row["revision"])
    run["warnings"] = warnings_for(run)
    return run


class Handler(BaseHTTPRequestHandler):
    server_version = "LayerAtlas/1"

    def identity(self):
        if self.server.trust_proxy_identity:
            login = self.headers.get("Tailscale-User-Login", "").strip()
            if login and len(login) <= 320 and not any(ord(c) < 32 for c in login):
                return {"actor": login, "identity_source": "tailscale"}
        return {"actor": "Unknown visitor", "identity_source": "unknown"}

    def record_activity(self, db, action, target, details):
        db.execute("""INSERT INTO activity(created_at,actor,identity_source,action,target,details,request_id)
                      VALUES(?,?,?,?,?,?,?)""", (
            datetime.now(timezone.utc).isoformat(), self.actor["actor"],
            self.actor["identity_source"], action, str(target),
            json.dumps(details, ensure_ascii=False), self.request_id))

    def log_message(self, format, *args):
        # send_bytes emits structured request records.
        pass

    def log_response(self, status):
        path = urlsplit(self.path).path
        if path != "/api/health":
            record = {"created_at": datetime.now(timezone.utc).isoformat(), "request_id": self.request_id,
                      **self.actor, "method": self.command, "path": path[:512],
                      "status": status, "duration_ms": round((time.monotonic() - self.started) * 1000, 2)}
            LOGGER.info(json.dumps(record, ensure_ascii=True))
            if status >= 400:
                try:
                    with self.db_connection() as db:
                        self.record_activity(db, "request.failed", path[:512], {"method": self.command, "status": status})
                except Exception:
                    LOGGER.exception("Could not persist failed request: %s", self.request_id)

    def send_bytes(self, status, body, content_type="application/json; charset=utf-8", filename=None):
        self.log_response(status)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", self.request_id)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    @contextmanager
    def db_connection(self):
        with closing(sqlite3.connect(self.server.db_path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def json_body(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            raise RequestError(415, "Use application/json for changes.")
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != self.headers.get("Host", "").lower() or parsed.path or parsed.query or parsed.fragment:
                raise RequestError(403, "Changes must come from this app's origin.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise RequestError(403, "Cross-site changes are blocked.")
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(400, "Use a Content-Length request.")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise RequestError(400, "A valid Content-Length is required.")
        if length < 0:
            raise RequestError(400, "Content-Length must be positive.")
        if length > MAX_BODY:
            self.close_connection = True
            raise RequestError(413, "Imports may be at most 25 MiB.")
        try:
            value = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise RequestError(400, "The request must contain valid JSON with finite numbers.")
        if not isinstance(value, dict):
            raise RequestError(400, "The request must be a JSON object.")
        return value

    def dispatch(self):
        path = urlsplit(self.path).path
        if self.command == "GET" and path == "/api/health":
            return self.send_json(200, {"ok": True})
        if self.command == "GET" and path in ("/", "/index.html", "/app.js", "/math.mjs", "/style.css", "/example.json"):
            filename = "index.html" if path == "/" else path[1:]
            file_path = ROOT / ("examples" if filename == "example.json" else "static") / filename
            content_type = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json"}[file_path.suffix]
            return self.send_bytes(200, file_path.read_bytes(), content_type + "; charset=utf-8")
        with self.db_connection() as db:
            if path == "/api/activity" and self.command == "POST":
                value = self.json_body()
                if value.get("action") not in ("view.open", "maps.compare", "png.export", "demo.toggle"):
                    raise RequestError(400, "Unknown browser action.")
                details = value.get("details", {})
                if not isinstance(details, dict) or len(json.dumps(details)) > 1000:
                    raise RequestError(400, "Browser action details must be an object up to 1000 characters.")
                self.record_activity(db, value["action"], "browser", {**details, "browser_reported": True})
                db.commit()
                return self.send_json(201, {"ok": True})
            if path == "/api/runs":
                if self.command == "GET":
                    runs = [unpack(row) for row in db.execute("SELECT * FROM runs ORDER BY created_at DESC")]
                    for run in runs:
                        run.pop("matrix")
                    return self.send_json(200, runs)
                if self.command == "POST":
                    payload = validate_run(self.json_body())
                    identifier = str(uuid.uuid4())
                    created = datetime.now(timezone.utc).isoformat()
                    db.execute("INSERT INTO runs VALUES(?,?,?,?)", (identifier, created, 1, json.dumps(payload, allow_nan=False)))
                    row = db.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone()
                    self.record_activity(db, "run.import", identifier, {"name": payload["name"], "model": payload["model"]})
                    db.commit()
                    return self.send_json(201, unpack(row))
            if path == "/api/tasks" and self.command == "GET":
                tasks = [dict(row) for row in db.execute("SELECT * FROM tasks ORDER BY id")]
                for task in tasks:
                    task["done"] = bool(task["done"])
                return self.send_json(200, tasks)
            task_match = re.fullmatch(r"/api/tasks/(\d+)", path)
            if task_match and self.command == "PATCH":
                value = self.json_body()
                if type(value.get("done")) is not bool:
                    raise RequestError(400, "done must be true or false.")
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT done FROM tasks WHERE id=?", (int(task_match[1]),)).fetchone()
                cursor = db.execute("UPDATE tasks SET done=? WHERE id=?", (value["done"], int(task_match[1])))
                if not cursor.rowcount:
                    raise RequestError(404, "Task not found.")
                row = dict(db.execute("SELECT * FROM tasks WHERE id=?", (int(task_match[1]),)).fetchone())
                self.record_activity(db, "task.update", task_match[1], {"title": row["title"], "before": bool(previous[0]), "done": value["done"]})
                db.commit()
                row["done"] = bool(row["done"])
                return self.send_json(200, row)
            run_match = re.fullmatch(r"/api/runs/([a-f0-9-]+)(?:/(export\.json|matrix\.csv))?", path)
            if run_match:
                identifier, download = run_match.groups()
                row = db.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone()
                if row is None:
                    raise RequestError(404, "Run not found.")
                run = unpack(row)
                if self.command == "GET":
                    self.record_activity(db, "run.export" if download else "run.view", identifier, {"name": run["name"], "format": download})
                    db.commit()
                    if download == "export.json":
                        payload = json.loads(row["payload"])
                        return self.send_bytes(200, json.dumps(payload, ensure_ascii=False, indent=2).encode(), filename="layer-atlas-run.json")
                    if download == "matrix.csv":
                        output = io.StringIO()
                        writer = csv.writer(output)
                        writer.writerow(["label"] + run["labels"])
                        writer.writerows([label] + values for label, values in zip(run["labels"], run["matrix"]))
                        return self.send_bytes(200, output.getvalue().encode(), "text/csv; charset=utf-8", "layer-atlas-matrix.csv")
                    return self.send_json(200, run)
                if download:
                    raise RequestError(405, "Downloads support GET only.")
                if self.command == "PATCH":
                    value = self.json_body()
                    if not isinstance(value.get("notes"), str) or len(value["notes"]) > 20000 or type(value.get("revision")) is not int:
                        raise RequestError(400, "Supply notes text (up to 20000 characters) and its integer revision.")
                    payload = json.loads(row["payload"])
                    payload["notes"] = value["notes"]
                    cursor = db.execute("UPDATE runs SET payload=?, revision=revision+1 WHERE id=? AND revision=?", (json.dumps(payload), identifier, value["revision"]))
                    if not cursor.rowcount:
                        raise RequestError(409, "Notes changed in another session. Reload the run before saving again.")
                    updated = db.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone()
                    self.record_activity(db, "notes.update", identifier, {"name": run["name"], "from_revision": run["revision"], "revision": run["revision"] + 1, "characters": len(value["notes"])})
                    db.commit()
                    return self.send_json(200, unpack(updated))
                if self.command == "DELETE":
                    self.json_body()
                    self.record_activity(db, "run.delete", identifier, {"name": run["name"]})
                    db.execute("DELETE FROM runs WHERE id=?", (identifier,))
                    db.commit()
                    return self.send_json(200, {"deleted": identifier})
        raise RequestError(404, "Page or API route not found.")

    def handle_request(self):
        self.request_id = str(uuid.uuid4())
        self.actor = self.identity()
        self.started = time.monotonic()
        try:
            try:
                host = urlsplit("//" + self.headers.get("Host", "")).hostname
            except ValueError:
                host = None
            if host not in self.server.allowed_hosts:
                raise RequestError(403, "This Host is not allowed.")
            self.dispatch()
        except RequestError as error:
            self.send_json(error.status, {"error": str(error)})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            LOGGER.exception("Request failed: %s", self.request_id)
            self.send_json(500, {"error": "The request failed. Check the server log."})

    do_GET = handle_request
    do_POST = handle_request
    do_PATCH = handle_request
    do_DELETE = handle_request


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def make_server(port=8786, db_path=None, allowed_hosts=(), bind="127.0.0.1", trust_proxy_identity=False):
    db_path = str(db_path or ROOT / "data" / "layer-atlas.sqlite3")
    initialize(db_path)
    server = ThreadingHTTPServer((bind, port), Handler)
    server.trust_proxy_identity = trust_proxy_identity
    server.db_path = db_path
    server.allowed_hosts = {"127.0.0.1", "localhost", *allowed_hosts}
    return server


def print_activity(db_path, limit, output=None):
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT created_at, actor, identity_source, action, target, details, request_id FROM activity ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    for created_at, actor, source, action, target, details, request_id in reversed(rows):
        print(f"{created_at}  {actor} ({source})  {action}  {target}  {details}  {request_id}", file=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--bind", default="127.0.0.1", help="Listen address; use 0.0.0.0 only inside the isolated container")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "layer-atlas.sqlite3")
    parser.add_argument("--allowed-host", action="append", default=[], help="Additional reverse-proxy hostname, without scheme or port")
    parser.add_argument("--trust-proxy-identity", action="store_true", help="Trust Tailscale identity headers; only behind the private Serve gateway")
    parser.add_argument("--activity", type=int, metavar="N", help="Print the newest N activity entries, oldest first, and exit")
    args = parser.parse_args()
    if args.activity is not None:
        if args.activity < 1:
            parser.error("--activity must be at least 1")
        print_activity(args.db, args.activity)
        raise SystemExit(0)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = make_server(args.port, args.db, args.allowed_host, args.bind, args.trust_proxy_identity)
    print(f"Layer Atlas serving at http://{args.bind}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
