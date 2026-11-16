# Layer Atlas

A workbench for the Transformer Layers and Attention-Head Similarity course project. Import computed layer or head maps from Colab, inspect cells, compare map structure, save notes, export figures and track the twelve team tasks.

## Run in Docker

Run these commands from the repository root with Docker Desktop running:

```sh
docker compose -f app/compose.yaml up -d --build --wait
```

Open <http://127.0.0.1:8786> or the private Tailscale URL below. SQLite lives in the Docker volume `layer-atlas_results`, not in the source checkout. Do not use `docker compose down -v` on the live project: that deletes the database volume.

The Python app runs as UID 10001 with a read-only root filesystem, no Linux capabilities, no new privileges, 1 CPU, 512 MiB RAM and 64 processes/threads. Only `/data` is persistent and writable; `/tmp` is bounded scratch space. The app is on an internal Docker network with no outbound internet route. It has no home-directory, Docker-socket or source-checkout mounts.

A separate non-root Nginx gateway publishes only `127.0.0.1:8786`. It has no database or host mounts, a read-only filesystem, a 0.5 CPU / 192 MiB / 32-process limit, and bounded temporary storage. It applies a shared 20 requests/second limit (burst 60), 16 simultaneous requests, a 25 MiB upload limit and request timeouts. HTTP 429 means wait briefly and retry. The gateway has outbound networking; isolation is not a guarantee against every container/runtime vulnerability. Keep Docker Desktop and the pinned base images updated.

For local development without Docker, use `python3 app/server.py --port 18786 --db /tmp/layer-atlas-dev.sqlite3`. Its default bind address remains loopback. The `--bind 0.0.0.0` option is for the isolated container only.

## Tailscale and sharing

The workbench is available at <https://macbook-pro-2.tailaedfae.ts.net:8443>. Tailscale Serve proxies that endpoint to the local gateway:

```sh
tailscale serve --bg --https=8443 http://127.0.0.1:8786
```

The saved tailnet policy permits external shared users (`autogroup:shared`) only TCP 8443 on this Mac's Tailscale IPv4 and IPv6 addresses. Tailnet members and tagged devices retain their previous access. The existing port 443 routes are unchanged and excluded from external shared-user access. No teammate invitation has been created by this deployment.

To give teammates access, share this machine individually through the Tailscale Machines page. Have them accept using their own account and connect their Tailscale client. Use **device sharing**, not an invitation to join your tailnet: tailnet members have broader access under the current policy. The saved rules have been validated by Tailscale; access from an external teammate's device still needs an end-to-end check. Revisit the destination addresses if this Mac is removed and re-enrolled.

Anyone with app access can download all results, import results, edit notes/tasks and delete runs. The app has no per-user roles. Network restrictions do not prevent an authorized user from deleting data.

### Installed service on this Mac

The user LaunchAgent `com.gherghina.layer-atlas` at `~/Library/LaunchAgents/com.gherghina.layer-atlas.plist` now runs `app/manage.py maintain` at login and every five minutes. It starts Docker Desktop if needed, ensures the built containers are running and healthy, and makes one backup per UTC day. Docker also restarts containers after process exits. Keep this Mac logged in and connected to Tailscale. First startup after a full reboot has not been exercised.

The separate `com.gherghina.layer-atlas-awake` LaunchAgent runs `app/awake.py` at login and restarts the watcher if it exits. While both app containers are running, it holds `caffeinate -is -w <watcher PID>`, matching Jobs Manager: idle sleep is prevented on battery or power; the stronger system-sleep assertion applies on power. Display sleep and screen locking remain allowed. Every 30 seconds it checks the containers and releases its assertion if either stops or Docker becomes unavailable. It does not override lid-close sleep, shutdown or loss of power. Plug in the Mac for sustained hosting. Logs are in `app/data/awake.log` and `app/data/awake-error.log`.

Inspect sleep prevention with `pmset -g assertions` and `launchctl print gui/501/com.gherghina.layer-atlas-awake`. To disable this app's keep-awake watcher, run `launchctl bootout gui/501/com.gherghina.layer-atlas-awake`; to restore it, run `launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.gherghina.layer-atlas-awake.plist`. Other apps such as Jobs Manager may independently prevent sleep.

Inspect the service and app:

```sh
launchctl print gui/501/com.gherghina.layer-atlas
docker compose -f app/compose.yaml ps
docker compose -f app/compose.yaml logs --tail 50
tailscale serve status
```

After changing source, rebuild with `docker compose -f app/compose.yaml up -d --build --wait`. Restarting alone does not copy source changes into an image. Host maintenance errors are in `app/data/maintenance-error.log`; backup events are in `app/data/maintenance.log`. Container logs are capped at three 5 MiB files per container.

To stop the app persistently, unload maintenance first, then stop containers:

```sh
launchctl bootout gui/501 ~/Library/LaunchAgents/com.gherghina.layer-atlas.plist
docker compose -f app/compose.yaml stop
```

Start it again with `launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.gherghina.layer-atlas.plist`. To stop only remote serving, use `tailscale serve --https=8443 off`.

### Backups and recovery

Backups use SQLite's online backup API and pass an integrity/schema check before being saved. They are stored on this Mac at `app/data/backups/layer-atlas-YYYY-MM-DD.sqlite3`, outside both containers, with private file permissions. The newest 30 daily snapshots are retained. A missed backup is retried on the next maintenance run. While the Mac is asleep/off, neither the app nor backups run. These local snapshots protect against app deletion, not loss of the Mac or Docker host compromise; copy them to independent storage for disaster recovery.

Make or refresh today's snapshot manually:

```sh
python3 app/manage.py backup
```

A pre-migration database snapshot and the old LaunchAgent plist are retained under `app/data/`. The old `app/data/layer-atlas.sqlite3` is an inactive pre-Docker copy, not the live database.

To restore a daily snapshot, first make a current backup, copy the selected snapshot to a separate file, unload maintenance, and stop both containers with the commands above. Verify the selected file with `manage.verify_backup(Path(...))`. With the app stopped, use a one-off `docker compose run --rm -T --no-deps atlas` process to replace `/data/layer-atlas.sqlite3` from that file and remove its stale `-wal` and `-shm` files. Start the containers and check results before reloading maintenance. The disposable container acceptance test exercises this exact recovery sequence, including restoration of deleted notes/results and changed tasks.

## Import from Colab

Each JSON file contains one computed symmetric matrix. This example assumes `matrix` already contains your similarity values as a NumPy array. Computing representations and similarity remains in the project pipeline.

```python
import json
from google.colab import files

result = {
    "schema_version": 1,
    "name": "Pythia-410M LAMBADA cosine",
    "model": "EleutherAI/pythia-410m",
    "task": "LAMBADA",
    "split": "test",
    "kind": "layer",
    "metric": "cosine",
    "pooling": "masked_mean",
    "sample_count": 400,
    "seed": 42,
    "max_tokens": 256,
    "accuracy": None,
    "chance_accuracy": None,
    "labels": [f"L{i}" for i in range(matrix.shape[0])],
    "matrix": matrix.tolist(),
    "notes": "",
    "provenance": "Record notebook commit, dataset revision and input formatting here.",
}
with open("result.json", "w", encoding="utf-8") as output:
    json.dump(result, output, allow_nan=False)
files.download("result.json")
```

Choose **Import result** in Layer Atlas. The downloadable `examples/example.json` contains clearly identified invented values for trying the format.

The required fields are `schema_version`, `name`, `model`, `task`, `kind`, `metric`, `pooling`, `sample_count`, `labels` and `matrix`. Other fields above are optional. Accuracy and chance accuracy are fractions between 0 and 1. `kind` is `layer` or `head`. For heads, use explicit unique labels such as `L0 H0`.

Supported metrics are `cosine`, `centered_cosine`, `linear_cka` and `js_divergence`. Cosine values range from −1 to 1. CKA and base-2 Jensen–Shannon divergence range from 0 to 1. Matrices must be symmetric within 1e-6, contain finite numbers, and have 2–512 unique labels. Each upload may be at most 25 MiB. Diagonal discrepancies produce a visible warning. Import metadata cannot mark a run as a built-in demo.

## Compare and export

Maps require the same kind and ordered labels for numerical comparison. The app computes Pearson correlation on the strict upper triangle, excluding the diagonal. Constant maps and maps with fewer than two off-diagonal pairs have undefined correlation. A−B requires matching metrics. Differences in model, task, pooling, sampling and truncation remain visible in the context table.

Download PNG exports the current map with its title, metadata, labels and color scale. JSON downloads preserve the portable import format. CSV downloads contain the labeled matrix. CSV labels are preserved verbatim, so spreadsheet users should import label columns as text when handling external labels.

**Explore demo maps** displays deterministic invented data separately from stored results. Its maps carry a synthetic label, including in PNG exports. Task completion is stored independently and starts unchecked. The task dates come from the project README as of 16 Nov 2026.

## Verify

```sh
python3 -m unittest discover -s app/tests -p 'test_*.py' -v
node app/tests/math.test.mjs
node --check app/static/app.js
```

Docker acceptance uses a separate disposable project and port; it never touches the live volume:

```sh
LAYER_ATLAS_PORT=18786 docker compose -p layer-atlas-verify -f app/compose.yaml up -d --no-build --wait
python3 app/tests/container_acceptance.py
docker compose -p layer-atlas-verify -f app/compose.yaml down -v
```

## Activity logging

Open **Activity** to see who imported, viewed, downloaded or deleted a result, edited notes, changed a task, compared maps or exported a PNG. Entries include UTC timestamps (displayed in your browser’s timezone), identity source, result/task identifiers and relevant change details. The newest 100 entries load first; **Load older activity** pages backward and **Refresh** fetches new events. Failed requests include their HTTP status. Note contents, matrices, cookies and request bodies are excluded from logs. Notes entries record revision changes and character counts; task entries record the previous and new completion state.

Activity is stored in the existing SQLite volume and included in database backups. Successful imports, edits and deletions commit together with their activity entry; if recording fails, the change rolls back. Deleting a result preserves its history. There is no app route to delete history. History starts with this update; earlier actions cannot be reconstructed. Database history currently has no automatic expiry.

The Docker app enables `--trust-proxy-identity` for the private gateway: remote requests use the `Tailscale-User-Login` supplied by [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve#identity-headers). Keep the backend private and the gateway published on loopback. Local processes that can reach the gateway are trusted and can supply these headers; do not put another public proxy in front of it without stripping client identity headers. Direct development servers ignore these headers by default. Tagged devices may have no user identity.

Without a Tailscale identity, use **Your name for local activity**. The name is stored in an HttpOnly, SameSite cookie and explicitly labeled **self-reported**, not authenticated. Visitors without either identity appear as **Unknown visitor**. Browser actions (view navigation, comparisons and PNG exports) are also self-reported, while imports, note edits, task changes, JSON/CSV downloads and deletions are recorded by the server. Browser telemetry does not confirm that a download was saved to disk. Everyone with app access can read history, including teammates’ Tailscale login names. This adds attribution, not authorization or tamper resistance against host/database administrators.

Each non-health HTTP request also emits a JSON container log with identity, method, path (without query), status, request ID and processing time. The same request ID links it to database history and the `X-Request-ID` response header. Health probes are omitted. Container log rotation retains the existing three 5 MiB files:

```sh
docker compose --env-file app/data/networks/layer-atlas.env -f app/compose.yaml logs -f atlas
```

## Import from Colab

Each JSON file contains one computed symmetric matrix. This example assumes `matrix` already contains your similarity values as a NumPy array. Computing representations and similarity remains in the project pipeline.

```python
import json
from google.colab import files

result = {
    "schema_version": 1,
    "name": "Pythia-410M LAMBADA cosine",
    "model": "EleutherAI/pythia-410m",
    "task": "LAMBADA",
    "split": "test",
    "kind": "layer",
    "metric": "cosine",
    "pooling": "masked_mean",
    "sample_count": 400,
    "seed": 42,
    "max_tokens": 256,
    "accuracy": None,
    "chance_accuracy": None,
    "labels": [f"L{i}" for i in range(matrix.shape[0])],
    "matrix": matrix.tolist(),
    "notes": "",
    "provenance": "Record notebook commit, dataset revision and input formatting here.",
}
with open("result.json", "w", encoding="utf-8") as output:
    json.dump(result, output, allow_nan=False)
files.download("result.json")
```

Choose **Import result** in Layer Atlas. The downloadable `examples/example.json` contains clearly identified invented values for trying the format.

The required fields are `schema_version`, `name`, `model`, `task`, `kind`, `metric`, `pooling`, `sample_count`, `labels` and `matrix`. Other fields above are optional. Accuracy and chance accuracy are fractions between 0 and 1. `kind` is `layer` or `head`. For heads, use explicit unique labels such as `L0 H0`.

Supported metrics are `cosine`, `centered_cosine`, `linear_cka` and `js_divergence`. Cosine values range from −1 to 1. CKA and base-2 Jensen–Shannon divergence range from 0 to 1. Matrices must be symmetric within 1e-6, contain finite numbers, and have 2–512 unique labels. Each upload may be at most 25 MiB. Diagonal discrepancies produce a visible warning. Import metadata cannot mark a run as a built-in demo.

## Compare and export

Maps require the same kind and ordered labels for numerical comparison. The app computes Pearson correlation on the strict upper triangle, excluding the diagonal. Constant maps and maps with fewer than two off-diagonal pairs have undefined correlation. A−B requires matching metrics. Differences in model, task, pooling, sampling and truncation remain visible in the context table.

Download PNG exports the current map with its title, metadata, labels and color scale. JSON downloads preserve the portable import format. CSV downloads contain the labeled matrix. CSV labels are preserved verbatim, so spreadsheet users should import label columns as text when handling external labels.

**Explore demo maps** displays deterministic invented data separately from stored results. Its maps carry a synthetic label, including in PNG exports. Task completion is stored independently and starts unchecked. The task dates come from the project README as of 16 Nov 2026.

## Verify

```sh
python3 -m unittest discover -s app/tests -p 'test_*.py' -v
node app/tests/math.test.mjs
node --check app/static/app.js
```

Docker acceptance uses a separate disposable project and port; it never touches the live volume:

```sh
python3 app/manage.py prepare --project layer-atlas-verify
LAYER_ATLAS_PORT=18786 docker compose --env-file app/data/networks/layer-atlas-verify.env -p layer-atlas-verify -f app/compose.yaml up -d --no-build --wait
python3 app/tests/container_acceptance.py
docker compose --env-file app/data/networks/layer-atlas-verify.env -p layer-atlas-verify -f app/compose.yaml down -v
docker network rm layer-atlas-verify_isolated
```
