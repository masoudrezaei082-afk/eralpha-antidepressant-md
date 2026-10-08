"""Run the production replicas on SaladCloud through a Salad job queue, with inputs, checkpoints and results on
Google Drive.

    python jobs/submit.py manifest              # (re)write jobs/manifest.yaml from prep/systems.yaml
    python jobs/submit.py inputs [--src DIR]    # upload built inputs to DRIVE_REMOTE/inputs/<system>/<pose>/
    python jobs/submit.py queue                 # create the Salad job queue (once, before `group`)
    python jobs/submit.py group                 # create the container group on that queue (stopped, 0 replicas)
    python jobs/submit.py secrets               # after `rclone config reconnect gdrive:`: new token to the group
    python jobs/submit.py submit [--only SYS..] # queue every replica that is not DONE and not already queued
    python jobs/submit.py replicas N            # set the group's replica count (0 stops it) and start it
    python jobs/submit.py image [--image REF]   # point the group at the pinned image (after a rebuild)
    python jobs/submit.py status                # per replica and node: stage, ns, speed, checkpoint check; budget
    python jobs/submit.py dashboard [--port P]  # the same, live in the browser on http://127.0.0.1:8765
    python jobs/submit.py length NS             # production length of every replica, running ones included
    python jobs/submit.py watch [--max N]       # requeue failed replicas; keep the group at (pending + running)
                                                # GPUs; stop it when done
    python jobs/submit.py cancel [--only SYS..] # cancel pending jobs (a running job is not stopped by this)

One job is one whole replica: the image's job_server.py runs run_replica.sh for it and answers when it exits. Salad
tries a job at most 4 times, and a node taken away mid-job counts as one try, so `watch` requeues a replica whose
job failed (or ended without state/DONE on Drive), up to RESUBMITS times; it resumes from its checkpoint on Drive.
The queue connection of a container group is fixed when the group is created.

Every command takes --dry-run (print the requests, send nothing). Credentials come from the environment or a
.env file in the repo root (see .env.example); nothing is written to disk or put in a job definition. Drive is
reached through rclone: DRIVE_REMOTE names a configured remote and folder (default gdrive:eralpha-antidepressant-md).

Drive layout (one folder per replica; the replica numbering k = 1..3 is per system, poses take contiguous blocks):
    inputs/<system>/<pose>/                  topol.top conf.gro index.ndx posre_*.itp     (written once, here)
    runs/<system>/<pose>/rep<k>/state/       what a resume needs: *.done, stage .gro/.cpt, md.tpr, md.cpt,
                                             md.prev.cpt, nodes.txt, perf.tsv, progress.json, DONE (pulled on resume)
    runs/<system>/<pose>/rep<k>/data/        md.partNNNN.{xtc,trr,edr,log,gro}, equilibration edr/log/mdp
                                             (push only: a resumed job never downloads trajectory)
    control/prod_ns                          production length in ns, read by every replica before each segment
"""
import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "jobs" / "manifest.yaml"
SALAD = "https://api.salad.com/api/public"
IMAGE = "ghcr.io/masoudrezaei082-afk/eralpha-antidepressant-md:2026.4-cu129-f0cf319"   # build run 37563518946
GROUP = "eralpha-antidepressant-md-q"                # the earlier Kelpie group "eralpha-antidepressant-md" cannot take a queue connection
QUEUE = "eralpha-antidepressant-md"
JOB_PORT = 8000                       # docker/job_server.py
# Salad GPU class name prefixes, comma-separated in GPU_CLASSES. The default matches "RTX 4070 Ti" and
# "RTX 4070 Ti Super" (both sm_89); GPU_CLASSES="RTX 5090" runs on the image's sm_120 code instead.
GPU_NAMES = tuple(s.strip() for s in os.environ.get("GPU_CLASSES", "RTX 4070 Ti").split(",") if s.strip())
PRIORITY = "medium"
CPU, MEMORY_MB = 8, 16384
RESUBMITS = 3                         # `watch` requeues a replica at most this often (each job: up to 4 tries)
PRICE_PER_H = float(os.environ.get("PRICE_PER_H", "0.17"))   # 4070 Ti Super at medium: covers both classes
EQUIL_NS = 5.0                        # 0.5 ns NVT at 2 fs counts double, + 4 ns NPT at 4 fs, in production-ns terms
INPUT_FILES = ("topol.top", "conf.gro", "index.ndx")


# ------------------------------------------------------------------------------------------------ configuration
def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#") and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def need(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"missing environment variable(s): {', '.join(missing)} (set them in .env, see .env.example)")
    return [os.environ[n] for n in names]


def seed(system, rep):
    """Fixed per replica, so a resubmitted replica regenerates exactly the same velocities."""
    # "shokouh/" is the original working name of the project. It fixes the seeds that were run, so keep it.
    return int(hashlib.sha256(f"shokouh/{system}/rep{rep}".encode()).hexdigest()[:7], 16)


def build_manifest():
    cfg = yaml.safe_load(open(ROOT / "prep" / "systems.yaml"))
    reps = []
    for system, sc in cfg["systems"].items():
        k = 0
        for pose in sc["poses"]:
            for _ in range(pose["replicas"]):
                k += 1
                reps.append({"system": system, "pose": pose["id"], "rep": k, "seed": seed(system, k)})
        if k != cfg["replicas"]:
            sys.exit(f"{system}: poses add up to {k} replicas, systems.yaml says {cfg['replicas']}")
    return reps


def manifest():
    if not MANIFEST.exists():
        sys.exit("jobs/manifest.yaml missing: run `python jobs/submit.py manifest` first")
    return yaml.safe_load(open(MANIFEST))["replicas"]


def prefix(r):
    return f"runs/{r['system']}/{r['pose']}/rep{r['rep']}/"


def select(reps, only):
    return [r for r in reps if not only or r["system"] in only]


# ------------------------------------------------------------------------------------------------ clients
def remote():
    return os.environ.get("DRIVE_REMOTE") or "gdrive:eralpha-antidepressant-md"


def rclone(*args, dry=False, capture=False, check=True):
    cmd = ["rclone", "--retries", "5", "--low-level-retries", "20", *args]
    if dry:
        print("would run: " + " ".join(cmd))
        return ""
    r = subprocess.run(cmd, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"{' '.join(cmd)} failed (exit {r.returncode}){': ' + r.stderr[-500:] if capture else ''}")
    return r.stdout if capture else r.returncode


def rclone_env():
    """The remote's config as RCLONE_CONFIG_<NAME>_<KEY> variables, for the container group's secrets."""
    name = remote().split(":", 1)[0]
    cfg = json.loads(rclone("config", "dump", capture=True)).get(name)
    if not cfg:
        sys.exit(f"no rclone remote '{name}': create it first (docs/cloud_run.md, one-time setup)")
    if cfg.get("type") != "drive" or cfg.get("scope") != "drive.file" or not cfg.get("client_id"):
        sys.exit(f"remote '{name}' must be type drive, scope drive.file, with its own client_id (docs/cloud_run.md)")
    return {f"RCLONE_CONFIG_{name.upper()}_{k.upper()}": str(v) for k, v in cfg.items()}


def http(method, url, dry, body=None):
    if dry and method != "GET":
        print(f"--- {method} {url}\n{json.dumps(body, indent=2)}")
        return {}
    import requests
    headers = {"Salad-Api-Key": need("SALAD_API_KEY")[0]}
    if body is not None:   # Salad updates are JSON merge patches
        headers["Content-Type"] = "application/merge-patch+json" if method == "PATCH" else "application/json"
    r = requests.request(method, url, headers=headers, data=None if body is None else json.dumps(body), timeout=60)
    if r.status_code >= 400:
        sys.exit(f"{method} {url} -> HTTP {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else {}


def salad_url(path="", kind="containers"):
    org, proj = os.environ.get("SALAD_ORG"), os.environ.get("SALAD_PROJECT")
    if not (org and proj):
        _, org, proj = need("SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT")
    return f"{SALAD}/organizations/{org}/projects/{proj}/{kind}{path}"


def queue_url(path=""):
    return salad_url(f"/{QUEUE}{path}", kind="queues")


def queue_jobs():
    """Every job ever put on the queue, oldest first (the API returns them unordered)."""
    jobs, page = [], 1
    while True:
        items = http("GET", queue_url(f"/jobs?page={page}&page_size=25"), False).get("items", [])   # 25 at most
        jobs += items
        if len(items) < 25:
            return sorted(jobs, key=lambda j: j["create_time"])
        page += 1


def job_key(job):
    i = job.get("input") or {}
    return (str(i.get("system")), str(i.get("pose")), str(i.get("rep")))


def by_replica(jobs):
    """{(system, pose, rep): [its jobs, oldest first]}"""
    out = {}
    for j in jobs:
        out.setdefault(job_key(j), []).append(j)
    return out


def event_time(e):
    return calendar.timegm(time.strptime(e["time"][:19], "%Y-%m-%dT%H:%M:%S"))


def job_hours(j, now):
    """Time the job spent with a node: from each `started` to the event after it (or now, while it runs)."""
    ev = sorted(j.get("events") or [], key=lambda e: e["time"])
    h = 0.0
    for k, e in enumerate(ev):
        if e["action"] == "started":
            end = event_time(ev[k + 1]) if k + 1 < len(ev) else (now if j["status"] == "running" else event_time(e))
            h += max(0.0, end - event_time(e)) / 3600
    return h


def attempts(j):
    return sum(1 for e in j.get("events") or [] if e["action"] == "started")


# ------------------------------------------------------------------------------------------------ commands
def cmd_manifest(a):
    reps = build_manifest()
    MANIFEST.write_text("# Generated by jobs/submit.py manifest from prep/systems.yaml; do not edit.\n"
                        "# seed = first 28 bits of sha256('shokouh/<system>/rep<k>'), used as gen-seed in nvt.mdp.\n"
                        + yaml.safe_dump({"replicas": reps}, sort_keys=False, width=120))
    print(f"jobs/manifest.yaml: {len(reps)} replicas, {len({r['system'] for r in reps})} systems")


def cmd_inputs(a):
    src = Path(a.src or ROOT / "systems")
    pairs = sorted({(r["system"], r["pose"]) for r in select(manifest(), a.only)})
    for system, pose in pairs:
        d = src / system / pose / "build"
        missing = [f for f in INPUT_FILES if not (d / f).is_file()]
        if missing:
            sys.exit(f"{d}: missing {', '.join(missing)} (build it first)")
        # --checksum: unchanged files (same MD5 on Drive) are skipped, so re-running is cheap
        rclone("copy", str(d), f"{remote()}/inputs/{system}/{pose}", "--checksum", "--transfers", "1",
               *[x for f in INPUT_FILES + ("posre_*.itp",) for x in ("--include", f)], dry=a.dry_run)
    print(f"inputs for {len(pairs)} system/pose pairs are in {remote()}/inputs/")


def cmd_queue(a):
    res = http("POST", salad_url(kind="queues"), a.dry_run, {
        "name": QUEUE, "display_name": "eralpha MD replicas",
        "description": "One job per ERa replica: {system, pose, rep, seed} for run_replica.sh"})
    if res:
        print(f"job queue {res.get('name', QUEUE)} created; next: `group`")


def cmd_group(a):
    try:
        drive_env = rclone_env()
    except SystemExit:
        if not a.dry_run:
            raise
        drive_env = {"RCLONE_CONFIG_GDRIVE_TYPE": "drive", "RCLONE_CONFIG_GDRIVE_TOKEN": "<token>"}
    if a.dry_run and not os.environ.get("SALAD_API_KEY"):
        classes = [{"name": n, "id": f"<{n} class id>", "prices": []} for n in GPU_NAMES]
    else:
        _, org, _ = need("SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT")
        classes = http("GET", f"{SALAD}/organizations/{org}/gpu-classes", False)["items"]
    chosen = [c for c in classes if any(c["name"].startswith(n) for n in GPU_NAMES)]
    if not chosen:
        sys.exit(f"no GPU class named like {GPU_NAMES}; available: {', '.join(c['name'] for c in classes)}")
    for c in chosen:
        price = next((p["price"] for p in c["prices"] if p["priority"] == PRIORITY), "?")
        print(f"GPU class {c['name']} ({c['id']}): ${price}/h at {PRIORITY} priority, "
              f"vCPU {c.get('min_vcpu')}-{c.get('max_vcpu')}, RAM {c.get('min_ram')}-{c.get('max_ram')} MB")
    cpu = min([CPU] + [c["max_vcpu"] for c in chosen if c.get("max_vcpu")])
    mem = min([MEMORY_MB] + [c["max_ram"] for c in chosen if c.get("max_ram")])
    body = {
        "name": GROUP, "display_name": "eralpha MD, ERa replicas, job queue",
        "replicas": 0, "autostart_policy": False, "restart_policy": "always",
        # fixed for the life of the group: the worker in the image takes jobs from QUEUE and POSTs them to :8000
        "queue_connection": {"path": "/", "port": JOB_PORT, "queue_name": QUEUE},
        # no job reaches an instance before job_server.py listens (a refused job would use up one of its 4 tries)
        "readiness_probe": {"tcp": {"port": JOB_PORT}, "initial_delay_seconds": 5, "period_seconds": 10,
                            "timeout_seconds": 2, "success_threshold": 1, "failure_threshold": 3},
        "container": {
            "image": a.image, "image_caching": True, "priority": PRIORITY,
            "resources": {"cpu": cpu, "memory": mem, "gpu_classes": [c["id"] for c in chosen]},
            # the queue worker authenticates with the node's IMDS token, so no Salad key is needed here
            "environment_variables": drive_env | {"DRIVE_REMOTE": remote(), "NTOMP": str(cpu)},
        },
    }
    if os.environ.get("GHCR_TOKEN"):     # only needed while the GHCR package is private
        body["container"]["registry_authentication"] = {
            "basic": {"username": need("GHCR_USER")[0], "password": os.environ["GHCR_TOKEN"]}}
    if a.dry_run:
        mask(body["container"]["environment_variables"])
    res = http("POST", salad_url(), a.dry_run, body)
    if res:
        print(f"container group {res['name']} created (id {res['id']}) on queue {QUEUE}, stopped; "
              f"next: `submit`, then `replicas N` to start it")


def mask(env):
    """Hide secret values before a dry run prints them."""
    for k in env:
        if k.endswith(("_TOKEN", "_CLIENT_SECRET")):
            env[k] = "<secret>"


def cmd_secrets(a):
    """After `rclone config reconnect gdrive:` (the token of a Testing-status OAuth app lasts 7 days), copy the new
    token into the existing group. Salad then restarts its running instances; each resumes from its checkpoint."""
    # Salad replaces environment_variables as a whole on PATCH, so keep the group's other variables
    current = http("GET", salad_url(f"/{GROUP}"), False)["container"].get("environment_variables", {})
    env = current | rclone_env()
    if a.dry_run:
        mask(env)
    http("PATCH", salad_url(f"/{GROUP}"), a.dry_run, {"container": {"environment_variables": env}})
    print("Drive credentials updated on the container group; run `submit` to requeue any replica whose job failed")


def drive_files(sub, include):
    """Paths under DRIVE_REMOTE/<sub> matching the rclone filter `include` (one listing call)."""
    out = rclone("lsf", "-R", "--files-only", "--include", include, f"{remote()}/{sub}", capture=True, check=False)
    return set(out.split())


def rkey(r):
    return (r["system"], r["pose"], str(r["rep"]))


def enqueue(reps, dry):
    """Create the replicas' folders on Drive, then put one job per replica on the queue."""
    # Drive allows two folders with the same name, and replicas of one system starting together would each create
    # runs/<system>/<pose>/ at once. One rclone call creates every folder first, serially, so they never race.
    with tempfile.TemporaryDirectory() as tmp:
        for r in reps:
            for d in ("state", "data"):
                (Path(tmp) / prefix(r) / d).mkdir(parents=True, exist_ok=True)
        (Path(tmp) / "control").mkdir()
        if reps:
            rclone("copy", tmp, remote(), "--create-empty-src-dirs", "--transfers", "1", "--checkers", "1", dry=dry)
    for r in reps:
        res = http("POST", queue_url("/jobs"), dry, {
            "input": {"system": r["system"], "pose": r["pose"], "rep": r["rep"], "seed": r["seed"]},
            "metadata": {"replica": prefix(r)[5:-1]}})
        if res:
            print(f"queued {prefix(r)[5:-1]} (job {res['id']})")


def cmd_submit(a):
    reps = select(manifest(), a.only)
    live = {} if a.dry_run else {k for k, js in by_replica(queue_jobs()).items()
                                 if js[-1]["status"] in ("pending", "running")}
    done = drive_files("runs", "DONE")                     # read-only, so also in a dry run
    # a replica counts as finished only by state/DONE on Drive: a succeeded job without it is queued again
    todo = [r for r in reps if rkey(r) not in live and prefix(r)[5:] + "state/DONE" not in done]
    print(f"{len(todo)} replica job(s) to queue ({len(reps) - len(todo)} already done or queued)")
    enqueue(todo, a.dry_run)


def cmd_image(a):
    """Point the existing group at another image (default: the pinned IMAGE). Running instances restart on it and
    resume from their checkpoints."""
    http("PATCH", salad_url(f"/{GROUP}"), a.dry_run, {"container": {"image": a.image}})
    print(f"container group {GROUP} now runs {a.image}")


def cmd_replicas(a):
    if a.n == 0:
        http("POST", salad_url(f"/{GROUP}/stop"), a.dry_run, {})
        print("container group stopped")
        return
    http("PATCH", salad_url(f"/{GROUP}"), a.dry_run, {"replicas": a.n})
    http("POST", salad_url(f"/{GROUP}/start"), a.dry_run, {})
    print(f"container group started with {a.n} replica(s)")


def prod_ns():
    out = rclone("cat", f"{remote()}/control/prod_ns", capture=True, check=False).strip()
    if out:
        return float(out), "control/prod_ns"
    md = (ROOT / "mdp" / "md.mdp").read_text()
    val = lambda k: float(next(l.split("=")[1].split(";")[0] for l in md.splitlines() if l.split("=")[0].strip() == k))
    return val("nsteps") * val("dt") / 1000, "mdp/md.mdp"


EQUIL_STAGES = {"em": 0.0, "nvt": 1.0, "npt_posres_1000": 0.5, "npt_posres_500": 0.5, "npt_posres_100": 0.5,
                "npt_posres_10": 0.5, "npt": 2.0}      # production-ns equivalents; they add up to EQUIL_NS
STALE_MIN = 25    # a running replica rewrites progress.json at every checkpoint (10 min) and every stage


def collect(only=None):
    """Everything status and dashboard show, from the job queue, Salad and Drive (two rclone calls for all replicas).
    A source that fails is reported in `errors` and the rest is still filled in."""
    now, errors = time.time(), []

    def guard(what, fn, default):
        try:
            return fn()
        except (Exception, SystemExit) as e:   # http() and rclone() exit on errors; keep the other sources
            errors.append(f"{what}: {e}")
            return default

    reps = select(manifest(), only)
    jobs = guard("Salad job queue", lambda: by_replica(queue_jobs()), {})
    group = guard("Salad container group", lambda: http("GET", salad_url(f"/{GROUP}"), False), {})
    insts = guard("Salad instances", lambda: http("GET", salad_url(f"/{GROUP}/instances"), False), {})
    out = rclone("lsjson", "-R", "--files-only", "--hash", "--fast-list", f"{remote()}/runs", capture=True, check=False)
    if not out:
        errors.append("Drive: cannot list runs/ (token expired? `rclone config reconnect gdrive:`, then `secrets`)")
    by_rep, small = {}, {}
    for f in json.loads(out or "[]"):
        by_rep.setdefault(f["Path"].split("/state/")[0].split("/data/")[0], {})[f["Name"]] = f
    with tempfile.TemporaryDirectory() as tmp:
        rclone("copy", f"{remote()}/runs", tmp, "--fast-list", "--include", "*/state/{progress.json,perf.tsv,nodes.txt}",
               check=False, capture=True)
        for f in Path(tmp).glob("*/*/*/state/*"):
            small[(f.parent.parent.relative_to(tmp).as_posix(), f.name)] = f.read_text(errors="replace")
    length, length_src = guard("control/prod_ns", prod_ns, (0.0, "?"))

    rows, nodes_seen = [], {}
    for r in reps:
        key = prefix(r)[5:-1]
        js = jobs.get(rkey(r), [])
        j = js[-1] if js else {}                       # the latest job decides the state
        names = by_rep.get(key, {})
        try:
            pr = json.loads(small.get((key, "progress.json")) or "{}")
        except ValueError:
            pr = {}
        perf = [l.split("\t") for l in (small.get((key, "perf.tsv")) or "").splitlines() if l.strip()]
        nodes = [l.split("\t") for l in (small.get((key, "nodes.txt")) or "").splitlines() if l.strip()]
        for n in nodes:
            if len(n) >= 5:
                nodes_seen[n[1]] = n[4].split(",")[0].strip()
        done_stages = [s for s in EQUIL_STAGES if f"{s}.done" in names]
        ns = float(pr.get("ns") or 0) if "md.cpt" in names else 0.0
        state = "done" if "DONE" in names else j.get("status", "not queued")
        if state == "done":
            ns = max(ns, length)
        elif state == "succeeded":                    # ended without DONE on Drive (e.g. a rejected job)
            state = "incomplete"
        tries = sum(attempts(x) for x in js)
        failures = max(0, tries - (1 if j.get("status") in ("running", "succeeded") else 0))
        cpt = names.get("md.cpt", {})
        cpt_age = (now - calendar.timegm(time.strptime(cpt["ModTime"][:19], "%Y-%m-%dT%H:%M:%S"))) / 60 if cpt else None
        prog_age = (now - pr["epoch"]) / 60 if pr.get("epoch") else None
        drive_md5 = cpt.get("Hashes", {}).get("md5", "")
        md5_ok = bool(drive_md5) and drive_md5 == pr.get("cpt_md5")
        hours = sum(job_hours(x, now) for x in js)
        alerts = []
        if state == "running" and (prog_age is None or prog_age > STALE_MIN):
            alerts.append(f"no progress for {prog_age:.0f} min" if prog_age else "running, no progress.json yet")
        if pr.get("cpt_rejected"):
            alerts.append(f"{pr['cpt_rejected']} checkpoint(s) failed validation (the last good one is kept)")
        if cpt and pr.get("cpt_md5") and not md5_ok and (prog_age or 0) > 5:
            alerts.append("md.cpt on Drive is not the one the node validated")
        if pr.get("status") == "failed":
            alerts.append("stopped: " + (pr.get("note") or "see data/*.log"))
        if state == "failed":
            alerts.append("job failed after its tries: `watch` requeues it (it resumes); data/*.log says why")
        elif state == "incomplete":
            alerts.append("job ended without DONE on Drive: `watch` or `submit` queues it again")
        if failures:
            alerts.append(f"{failures} failed or interrupted tr{'y' if failures == 1 else 'ies'} over {len(js)} job(s)")
        if len(js) > RESUBMITS:
            alerts.append(f"queued {len(js)} times: `watch` stops requeueing it; read data/*.log")
        equil = sum(EQUIL_STAGES[s] for s in done_stages)
        rows.append({
            "replica": key, "system": r["system"], "pose": r["pose"], "rep": r["rep"], "state": state,
            "stage": "done" if state == "done" else pr.get("stage") or ("md" if len(done_stages) == 7 else "-"),
            "stages_done": len(done_stages), "ns": round(ns, 3), "target_ns": length,
            "pct": round(100 * (equil + min(ns, length)) / (EQUIL_NS + length), 1) if length else 0,
            "ns_per_day": pr.get("ns_per_day") or (float(perf[-1][-1]) if perf else None),
            "segments_ns_per_day": [float(p[-1]) for p in perf if len(p) >= 3],
            "machine": pr.get("machine") if state == "running" else None, "gpu": pr.get("gpu") or "",
            "gpu_util": pr.get("gpu_util"), "gpu_temp": pr.get("gpu_temp"),
            "nodes": [{"utc": n[0], "machine": n[1], "gpu": n[4].split(",")[0].strip() if len(n) >= 5 else ""}
                      for n in nodes],
            "cpt_step": pr.get("step"), "cpt_age_min": None if cpt_age is None else round(cpt_age, 1),
            "progress_age_min": None if prog_age is None else round(prog_age, 1),
            "cpt_md5_ok": md5_ok, "cpt_prev": "md.prev.cpt" in names, "cpt_rejected": pr.get("cpt_rejected", 0),
            "failures": failures, "jobs": len(js), "gpu_hours": round(hours, 2), "alerts": alerts,
            "parts": sum(1 for n in names if n.startswith("md.part") and n.endswith(".xtc")),
        })

    running_on = {x["machine"]: x["replica"] for x in rows if x["state"] == "running" and x["machine"]}
    instances = [{"machine": i.get("machine_id"), "state": i.get("state"), "ready": i.get("ready"),
                  "updated": i.get("update_time"), "cpu_percent": i.get("cpu_percent"),
                  "memory_mb": i.get("memory_usage_mb"), "pulling_pct": i.get("pulling_progress"),
                  "replica": running_on.get(i.get("machine_id")), "gpu": nodes_seen.get(i.get("machine_id"), "")}
                 for i in insts.get("instances", [])]

    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    speeds = sorted(x["ns_per_day"] for x in rows if x["ns_per_day"])
    med = speeds[len(speeds) // 2] if speeds else None
    total_eq = len(rows) * (EQUIL_NS + length)
    done_eq = sum(x["pct"] / 100 * (EQUIL_NS + length) for x in rows)
    gpu_h = sum(x["gpu_hours"] for x in rows)
    budget = float(os.environ.get("BUDGET_USD", "30"))
    n_all = len(manifest())
    active = sum(1 for i in instances if i["state"] == "running")
    summary = {
        "counts": counts, "replicas": len(rows), "length_ns": length, "length_src": length_src,
        "ns_secured": round(sum(x["ns"] for x in rows), 2), "ns_total": len(rows) * length,
        "pct": round(100 * done_eq / total_eq, 1) if total_eq else 0,
        "gpu_hours": round(gpu_h, 2), "spend_usd": round(gpu_h * PRICE_PER_H, 2), "price_per_h": PRICE_PER_H,
        "budget_usd": budget, "median_ns_per_day": med, "gpus_running": active,
        "group_status": (group.get("current_state") or {}).get("status"), "group_replicas": group.get("replicas"),
        "image": (group.get("container") or {}).get("image"),
    }
    alerts = []
    if med:
        summary["need_gpu_hours"] = round(n_all * (length + EQUIL_NS) / med * 24, 1)
        summary["need_usd"] = round(summary["need_gpu_hours"] * PRICE_PER_H, 2)
        summary["fit_ns"] = round((budget / 1.10 / PRICE_PER_H) / n_all * med / 24 - EQUIL_NS, 1)
        if active:
            summary["eta_h"] = round((total_eq - done_eq) / (med * min(active, len(rows))) * 24, 1)
        if summary["fit_ns"] < length:
            alerts.append(f"at {med:.0f} ns/day ${budget:g} fits {summary['fit_ns']:.0f} ns per replica, "
                          f"not {length:g}: run `length {summary['fit_ns']:.0f}`")
    if active and not counts.get("running") and not counts.get("pending"):
        alerts.append(f"{active} GPU(s) running with no job left: `replicas 0` stops the billing")
    if summary["image"] and summary["image"] != IMAGE:
        alerts.append(f"the group runs {summary['image']}, not the pinned {IMAGE}: `image` updates it")
    alerts += [f"{x['replica']}: {m}" for x in rows for m in x["alerts"]]
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "summary": summary, "replicas": rows,
            "instances": instances, "alerts": alerts, "errors": errors}


def cmd_status(a):
    d = collect(a.only)
    s = d["summary"]
    age = lambda m: "-" if m is None else f"{m:.0f}m"
    print(f"{'replica':<30} {'state':<10} {'stage':<16} {'ns':>13} {'%':>5} {'ns/day':>6} {'node':<22} "
          f"{'nodes':>5} {'ckpt':>5}  check")
    for x in d["replicas"]:
        node = f"{(x['machine'] or '')[:8]} {x['gpu'].replace('NVIDIA GeForce ', '')}".strip()
        check = ("md5 ok" if x["cpt_md5_ok"] else "md5 ?") + (" +prev" if x["cpt_prev"] else "") \
            if x["cpt_step"] else "-"
        if x["cpt_rejected"]:
            check += f" {x['cpt_rejected']} rejected"
        speed = f"{x['ns_per_day']:.0f}" if x["ns_per_day"] else ""
        print(f"{x['replica']:<30} {x['state']:<10} {x['stage']:<16} {x['ns']:>6.2f}/{x['target_ns']:<6g} "
              f"{x['pct']:>5.1f} {speed:>6} {node:<22} {len(x['nodes']):>5} {age(x['cpt_age_min']):>5}  {check}")
    print("  ".join(f"{k}: {v}" for k, v in sorted(s["counts"].items())))
    if d["instances"]:
        print(f"\n{'node (Salad machine)':<38} {'state':<11} {'ready':<5} {'cpu%':>5} {'GPU':<24} replica")
        for i in d["instances"]:
            print(f"{i['machine'] or '-':<38} {i['state'] or '-':<11} {str(i['ready']):<5} "
                  f"{i['cpu_percent'] or 0:>5.0f} {i['gpu']:<24} {i['replica'] or '-'}")
    print(f"\ngroup {s['group_status']} ({s['gpus_running']} GPU(s) running); production length {s['length_ns']:g} ns "
          f"({s['length_src']}); {s['ns_secured']:.1f} of {s['ns_total']:g} ns secured on Drive, {s['pct']:.1f}% of all "
          f"work; job time {s['gpu_hours']:.1f} GPU-h ≈ ${s['spend_usd']:.2f} at ${PRICE_PER_H}/h (idle GPUs not counted)")
    if s["median_ns_per_day"]:
        print(f"measured median {s['median_ns_per_day']:.0f} ns/day: {s['replicas']} replicas x {s['length_ns']:g} ns "
              f"need ≈ {s['need_gpu_hours']:.0f} GPU-h ≈ ${s['need_usd']:.2f}; ${s['budget_usd']:g} with a 10% reserve "
              f"fits {s['fit_ns']:.0f} ns per replica" + (f"; ≈ {s['eta_h']:.1f} h to go" if "eta_h" in s else ""))
    for m in d["alerts"]:
        print(f"ALERT {m}")
    for m in d["errors"]:
        print(f"ERROR {m}")


def cmd_dashboard(a):
    """Live page on http://127.0.0.1:PORT. Credentials stay in this process: the browser only gets collect()'s JSON,
    which this process refreshes every --every seconds (Drive, the job queue and Salad are read here, not in the page)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    page = (ROOT / "jobs" / "dashboard.html").read_bytes()
    cache, wake = {"body": b'{"loading": true}'}, threading.Event()

    def refresher():
        while True:
            t0 = time.time()
            try:
                d = collect()
            except (Exception, SystemExit) as e:   # keep serving the last good data
                d = json.loads(cache["body"]) | {"errors": [f"refresh failed: {e}"]}
            d["refresh_s"], d["every_s"] = round(time.time() - t0, 1), a.every
            cache["body"] = json.dumps(d).encode()
            wake.wait(a.every)
            wake.clear()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/status"):
                if "refresh=1" in self.path:
                    wake.set()
                body, ctype = cache["body"], "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    threading.Thread(target=refresher, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"dashboard on http://127.0.0.1:{a.port} (data refreshed every {a.every} s; Ctrl-C stops it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_length(a):
    if a.ns <= 0:
        sys.exit("length must be positive")
    if a.dry_run:
        print(f"would write {a.ns:g} to {remote()}/control/prod_ns")
        return
    subprocess.run(["rclone", "rcat", f"{remote()}/control/prod_ns"], input=f"{a.ns:g}\n", text=True, check=True)
    print(f"control/prod_ns = {a.ns:g} ns: every replica picks it up before its next segment (within an hour); one "
          f"already past the new length stops there and is DONE. Set endframe = {a.ns * 100:.0f} in analysis/mmpbsa_*.in.")


def cmd_watch(a):
    """Every 5 minutes: requeue each replica whose latest job failed or ended without state/DONE on Drive (at most
    RESUBMITS times; a cancelled job is left alone), then set the group to (pending + running) GPUs, at most --max,
    and stop it once nothing is pending or running. job_server.py gives an idle instance the lowest deletion cost,
    so a scale-down removes idle GPUs, not running replicas."""
    current, warned, errors = None, set(), 0
    reps = {rkey(r): r for r in manifest()}
    while True:
        try:                             # one failed API call (Salad answers 504 at times) must not end the watch
            jobs = by_replica(queue_jobs()) if not a.dry_run or os.environ.get("SALAD_API_KEY") else {}
            done = drive_files("runs", "DONE")
            again = []
            for k, js in jobs.items():
                r, last = reps.get(k), js[-1]["status"]
                if not r or prefix(r)[5:] + "state/DONE" in done or last not in ("failed", "succeeded"):
                    continue
                if len(js) > RESUBMITS:
                    if k not in warned:
                        print(f"{time.strftime('%H:%M')} ALERT {'/'.join(k)}: {len(js)} jobs ended without DONE; "
                              f"not requeued again, read its data/*.log and run `submit --only {k[0]}`")
                        warned.add(k)
                    continue
                again.append(r)
            if again:
                print(f"{time.strftime('%H:%M')} requeueing {len(again)}: {', '.join(prefix(r)[5:-1] for r in again)}")
                enqueue(again, a.dry_run)
            want = sum(1 for js in jobs.values() if js[-1]["status"] in ("pending", "running")) + len(again)
            target = min(want, a.max)
            if target != current:
                if target == 0:
                    http("POST", salad_url(f"/{GROUP}/stop"), a.dry_run, {})
                    print(f"{time.strftime('%H:%M')} no pending or running jobs: container group stopped")
                    return
                http("PATCH", salad_url(f"/{GROUP}"), a.dry_run, {"replicas": target})
                if current is None and (a.dry_run or http("GET", salad_url(f"/{GROUP}"), False)
                                        .get("current_state", {}).get("status") == "stopped"):
                    http("POST", salad_url(f"/{GROUP}/start"), a.dry_run, {})
                print(f"{time.strftime('%H:%M')} {want} job(s) pending or running: {target} GPU(s)")
                current = target
            errors = 0
        except (SystemExit, Exception) as e:   # API errors exit via http(); rclone ones raise
            errors += 1
            print(f"{time.strftime('%H:%M')} {'ALERT ' if errors >= 6 else ''}check failed ({errors} in a row), "
                  f"retrying in 1 min: {(str(e) or type(e).__name__).splitlines()[0][:200]}", flush=True)
            if a.dry_run:
                raise
            time.sleep(60)
            continue
        if a.dry_run:
            return
        time.sleep(300)


def cmd_cancel(a):
    """Cancels pending jobs. Salad does not stop a running job when it is cancelled: `replicas 0` does that (the
    replica pushes a checkpoint on SIGTERM and resumes from it when queued again)."""
    reps = {rkey(r) for r in select(manifest(), a.only)}
    for k, js in by_replica(queue_jobs()).items():
        j = js[-1]
        if k in reps and j["status"] in ("pending", "running"):
            http("DELETE", queue_url(f"/jobs/{j['id']}"), a.dry_run)
            print(f"cancelled {'/'.join(k)} (job {j['id']}, was {j['status']})")


def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="print requests instead of sending them")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("queue")
    s = sub.add_parser("inputs"); s.add_argument("--src", help="directory holding <system>/<pose>/build/")
    s.add_argument("--only", nargs="+")
    s = sub.add_parser("group"); s.add_argument("--image", default=IMAGE,
                                                help="image to run (default: the pinned build above)")
    sub.add_parser("secrets")
    s = sub.add_parser("image"); s.add_argument("--image", default=IMAGE)
    s = sub.add_parser("submit"); s.add_argument("--only", nargs="+")
    s = sub.add_parser("replicas"); s.add_argument("n", type=int)
    s = sub.add_parser("status"); s.add_argument("--only", nargs="+")
    s = sub.add_parser("dashboard"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--every", type=int, default=60, help="seconds between refreshes (default 60)")
    s = sub.add_parser("length"); s.add_argument("ns", type=float, help="production length per replica, in ns")
    s = sub.add_parser("watch"); s.add_argument("--max", type=int, default=9, help="most GPUs at once (default 9)")
    s = sub.add_parser("cancel"); s.add_argument("--only", nargs="+")
    a = p.parse_args()
    if not hasattr(a, "only"):
        a.only = None
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
