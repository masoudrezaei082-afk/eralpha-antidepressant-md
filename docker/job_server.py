#!/usr/bin/env python3
"""HTTP front for run_replica.sh, called by Salad's job queue worker (queue_entry.sh starts both).

    GET  /hc    200 once the server is up (the container group's readiness probe; the worker takes no job before)
    POST /      one job: {"system": S, "pose": P, "rep": K, "seed": N} -> run_replica.sh S P K N, in a fresh
                directory under $WORK, and answer when it exits. A replica takes hours; the worker waits without a
                timeout and keeps the job's lease alive meanwhile.

The worker counts 408, 429 and 5xx as a failed attempt (Salad retries a job up to 4 times) and stores any other
answer as the job's output. So: exit 0 (the replica is DONE on Drive) -> 200 with progress.json; a non-zero exit
-> 500; a malformed job -> 400, which ends it at once ("succeeded" with ok false) instead of failing 4 times;
a second job while one runs -> 503. jobs/submit.py trusts state/DONE on Drive, never the job status alone.

SIGTERM (Salad stopping the container) goes to run_replica.sh, which saves and pushes a checkpoint first.
On Salad, the instance's deletion cost is 1 while a job runs and 0 when idle, so a scale-down removes idle GPUs.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("JOB_PORT", "8000"))
WORK = Path(os.environ.get("WORK", "/work"))
IMDS = os.environ.get("IMDS_URL", "http://169.254.169.254")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
busy = threading.Lock()
child = {"proc": None}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] job_server: {msg}", flush=True)


def deletion_cost(value):
    """Best effort, in the background: IMDS exists only on Salad (SALAD_MACHINE_ID is set there)."""
    if not os.environ.get("SALAD_MACHINE_ID"):
        return

    def put():
        try:
            req = urllib.request.Request(f"{IMDS}/v1/deletion-cost", method="PUT",
                                         data=json.dumps({"deletion_cost": value}).encode(),
                                         headers={"Metadata": "true", "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            log(f"deletion cost {value} not set: {e}")
    threading.Thread(target=put, daemon=True).start()


def parse(body):
    job = json.loads(body or b"{}")
    args = [str(job.get(k, "")) for k in ("system", "pose", "rep", "seed")]
    if not (NAME.match(args[0]) and NAME.match(args[1]) and args[2].isdigit() and args[3].isdigit()):
        raise ValueError(f"need system, pose, rep, seed; got {job!r}"[:300])
    return args


def run(args, job_id):
    d = WORK / "job"                     # one job at a time; Drive holds everything a resume needs
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.time()
    log(f"job {job_id}: run_replica.sh {' '.join(args)}")
    # its own process group, so a SIGTERM reaches run_replica.sh (and through it mdrun) and nothing else
    proc = subprocess.Popen(["run_replica.sh", *args], cwd=d, start_new_session=True)
    child["proc"] = proc
    rc = proc.wait()
    child["proc"] = None
    rel = d / "runs" / args[0] / args[1] / f"rep{args[2]}" / "state" / "progress.json"
    try:
        progress = json.loads(rel.read_text())
    except (OSError, ValueError):
        progress = None
    shutil.rmtree(d, ignore_errors=True)
    out = {"ok": rc == 0, "exit": rc, "replica": "/".join(args[:3]), "hours": round((time.time() - t0) / 3600, 3),
           "machine": os.environ.get("SALAD_MACHINE_ID", "local"), "progress": progress}
    log(f"job {job_id}: exit {rc} after {out['hours']} h")
    return rc, out


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/hc":
            self.reply(200, {"ok": True, "busy": busy.locked()})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        job_id = self.headers.get("Salad-Job-Id", "-")
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            args = parse(body)
        except ValueError as e:
            log(f"job {job_id}: rejected: {e}")
            self.reply(400, {"ok": False, "error": str(e)})
            return
        if not busy.acquire(blocking=False):
            log(f"job {job_id}: rejected, another job is running")
            self.reply(503, {"ok": False, "error": "busy"})
            return
        try:
            deletion_cost(1)
            rc, out = run(args, job_id)
        finally:
            deletion_cost(0)
            busy.release()
        try:
            self.reply(200 if rc == 0 else 500, out)
        except OSError:                  # the worker went away (container stopping); the job state is on Drive
            pass

    def log_message(self, fmt, *a):
        if not self.path.startswith("/hc"):
            log(fmt % a)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True

    def on_term(signum, _):
        proc = child["proc"]
        if proc is None:
            sys.exit(0)
        log("SIGTERM: run_replica.sh saves a checkpoint, then stops")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        threading.Thread(target=lambda: (proc.wait(), os._exit(143)), daemon=True).start()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    deletion_cost(0)
    log(f"listening on :{PORT}, jobs run in {WORK / 'job'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
