import copy
from contextlib import closing
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location("layer_atlas_server", Path(__file__).parents[1] / "server.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "results.sqlite3"
        self.server = server_module.make_server(0, self.path, ["atlas.example"])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.example = json.loads((Path(__file__).parents[1] / "examples/example.json").read_text())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        data = json.dumps(payload) if payload is not None else None
        connection.request(method, path, data, request_headers)
        response = connection.getresponse()
        body = response.read()
        status = response.status
        connection.close()
        return status, body

    def test_import_notes_exports_and_tasks(self):
        status, body = self.request("POST", "/api/runs", self.example)
        self.assertEqual(status, 201)
        run = json.loads(body)
        identifier = run["id"]
        self.assertNotIn("demo", run)
        status, body = self.request("PATCH", f"/api/runs/{identifier}", {"notes": "Saved note", "revision": 1})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["revision"], 2)
        self.assertEqual(self.request("PATCH", f"/api/runs/{identifier}", {"notes": "Stale note", "revision": 1})[0], 409)
        status, body = self.request("GET", f"/api/runs/{identifier}/export.json")
        self.assertEqual(status, 200)
        exported = json.loads(body)
        self.assertEqual(exported["notes"], "Saved note")
        self.assertNotIn("id", exported)
        self.assertEqual(server_module.validate_run(exported)["matrix"], self.example["matrix"])
        status, body = self.request("GET", f"/api/runs/{identifier}/matrix.csv")
        self.assertEqual(status, 200)
        self.assertIn(b"label,L0,L1,L2", body)
        self.assertEqual(self.request("PATCH", "/api/tasks/1", {"done": True})[0], 200)
        with closing(server_module.sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT done FROM tasks WHERE id=1").fetchone()[0], 1)
            self.assertIn("Saved note", db.execute("SELECT payload FROM runs").fetchone()[0])
        self.assertEqual(self.request("DELETE", f"/api/runs/{identifier}", {})[0], 200)
        self.assertEqual(self.request("GET", f"/api/runs/{identifier}")[0], 404)

    def test_validation_and_failed_import_atomicity(self):
        invalid = [
            ("metric", []), ("sample_count", True), ("sample_count", 0),
            ("labels", ["L0", "L0", "L2"]), ("accuracy", 10**400),
            ("matrix", [[1, 10**400, .2], [10**400, 1, .8], [.2, .8, 1]]),
            ("matrix", [[1, True, .2], [True, 1, .8], [.2, .8, 1]]),
            ("matrix", [[1, .4, .2], [.5, 1, .8], [.2, .8, 1]]),
            ("name", "\ud800"),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=str(value)[:30]):
                payload = copy.deepcopy(self.example)
                payload[field] = value
                self.assertEqual(self.request("POST", "/api/runs", payload)[0], 400)
        self.assertEqual(json.loads(self.request("GET", "/api/runs")[1]), [])

    def test_proxy_origin_and_static_allowlist(self):
        self.assertEqual(self.request("POST", "/api/runs", self.example, {"Host": "atlas.example:8443", "Origin": "https://atlas.example:8443"})[0], 201)
        self.assertEqual(self.request("POST", "/api/runs", self.example, {"Origin": "https://external.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/runs", self.example, {"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/runs", self.example, {"Content-Type": "text/plain"})[0], 415)
        for path in ["/server.py", "/data/layer-atlas.sqlite3", "/../server.py"]:
            self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", "/")[0], 200)

    def test_activity_identity_and_deleted_run_history(self):
        headers = {"Cookie": "atlas_name=Bogdan", "Tailscale-User-Login": "spoof@example.com"}
        self.assertEqual(json.loads(self.request("GET", "/api/me", headers=headers)[1]),
                         {"actor": "Bogdan", "identity_source": "self-reported"})
        status, body = self.request("POST", "/api/runs", self.example, headers)
        identifier = json.loads(body)["id"]
        path = f"/api/runs/{identifier}"
        self.request("PATCH", path, {"notes": "private note text", "revision": 1}, headers)
        self.request("PATCH", "/api/tasks/1", {"done": True}, headers)
        self.request("GET", path + "/matrix.csv", headers=headers)
        self.request("DELETE", path, {}, headers)
        self.request("PATCH", path, {"notes": "missing", "revision": 2}, headers)
        events = json.loads(self.request("GET", "/api/activity")[1])
        self.assertEqual([e["action"] for e in events],
                         ["request.failed", "run.delete", "run.export", "task.update", "notes.update", "run.import"])
        self.assertTrue(all(e["actor"] == "Bogdan" for e in events))
        self.assertTrue(all(e["identity_source"] == "self-reported" for e in events))
        self.assertNotIn("private note text", json.dumps(events))
        self.assertEqual(events[1]["details"]["name"], self.example["name"])
        self.assertEqual(events[2]["details"]["format"], "matrix.csv")
        self.assertFalse(events[3]["details"]["before"])
        self.assertTrue(events[3]["details"]["done"])
        older = json.loads(self.request("GET", f'/api/activity?before={events[2]["id"]}')[1])
        self.assertEqual(older, events[3:])
        server_module.initialize(self.path)
        with closing(server_module.sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM activity").fetchone()[0], 6)

    def test_proxy_identity_and_local_name(self):
        self.server.trust_proxy_identity = True
        headers = {"Tailscale-User-Login": "teammate@example.com", "Cookie": "atlas_name=Other"}
        identity = json.loads(self.request("GET", "/api/me", headers=headers)[1])
        self.assertEqual(identity, {"actor": "teammate@example.com", "identity_source": "tailscale"})
        identity = json.loads(self.request("POST", "/api/me", {"name": "Changed"}, headers)[1])
        self.assertEqual(identity["actor"], "teammate@example.com")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", "/api/me", json.dumps({"name": "Bogdan G"}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        cookie = response.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertEqual(json.loads(response.read())["actor"], "Bogdan G")
        connection.close()
        identity = json.loads(self.request("GET", "/api/me", headers={"Cookie": cookie.split(";", 1)[0]})[1])
        self.assertEqual(identity["actor"], "Bogdan G")
        self.assertEqual(self.request("POST", "/api/me", {"name": "bad\\nname".replace("\\n", "\n")})[0], 400)
        self.assertEqual(json.loads(self.request("GET", "/api/me")[1])["identity_source"], "unknown")

    def test_audit_failure_rolls_back_change(self):
        with closing(server_module.sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON activity BEGIN SELECT RAISE(ABORT, 'test audit failure'); END")
        with self.assertLogs("layer_atlas", level="ERROR"):
            self.assertEqual(self.request("POST", "/api/runs", self.example)[0], 500)
        self.assertEqual(json.loads(self.request("GET", "/api/runs")[1]), [])

    def test_browser_events_validation_and_structured_request_log(self):
        with self.assertLogs("layer_atlas", level="INFO") as logs:
            self.request("POST", "/api/activity", {"action": "png.export", "details": {"id": "abc", "browser_reported": False}})
        record = json.loads(logs.records[0].message)
        self.assertEqual(record["status"], 201)
        self.assertIn("duration_ms", record)
        events = json.loads(self.request("GET", "/api/activity")[1])
        self.assertTrue(events[0]["details"]["browser_reported"])
        self.assertEqual(events[0]["request_id"], record["request_id"])
        self.assertEqual(self.request("POST", "/api/activity", {"action": "run.delete"})[0], 400)
        self.assertEqual(self.request("POST", "/api/activity", {"action": "png.export", "details": {"x": "a" * 1001}})[0], 400)
        for query in ("bad", "-1", str(2**70)):
            self.assertEqual(self.request("GET", "/api/activity?before=" + query)[0], 400)
        self.assertEqual(self.request("POST", "/api/activity", {"action": "view.open"}, {"Origin": "https://bad.example"})[0], 403)

    def test_oversized_body_rejected_before_read(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", "/api/runs", b"", {"Content-Type": "application/json", "Content-Length": str(server_module.MAX_BODY + 1)})
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        connection.close()


if __name__ == "__main__":
    unittest.main()
