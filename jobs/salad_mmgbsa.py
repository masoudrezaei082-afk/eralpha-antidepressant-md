#!/usr/bin/env python3
"""Run the MM/GBSA stage of the post-MD analysis on SaladCloud CPU-only nodes (no queue, no GPU).

    python jobs/salad_mmgbsa.py create KEY [KEY ...]   # one stopped group per key (system key or key_repN)
    python jobs/salad_mmgbsa.py start KEY [KEY ...]    # start them (1 instance each)
    python jobs/salad_mmgbsa.py status                 # groups and instances, finished jobs on Drive
    python jobs/salad_mmgbsa.py stop KEY [KEY ...]     # stop them (billing ends; the groups stay, so `start` resumes)

Each group runs the CPU analysis image (docker/analysis/Dockerfile) with MM_KEYS=<KEY>: its run_mmgbsa.sh waits
for each replica's processed trajectory on Drive, runs gb_eps2 (with per-residue decomposition), gb_eps1, gb_eps4
and pb_eps2 over all vCPUs, and copies each finished job to Drive mmgbsa-salad/<id>/<tag>/ with a DONE marker.
Jobs already DONE on Drive are skipped, so a preempted and reallocated node resumes. CPU-only nodes are billed per
vCPU-hour and GB-hour (16 vCPU + 16 GB at about $0.10/h); the container exits when its jobs are finished
(restart_policy never) and `status` says which groups can be stopped.

Credentials as for jobs/submit.py: SALAD_API_KEY/SALAD_ORG/SALAD_PROJECT from .env, the Drive token from the rclone
remote into the group's environment (never printed, never in the image).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit as S  # noqa: E402

IMAGE = os.environ.get("ANA_IMAGE", "ghcr.io/masoudrezaei082-afk/eralpha-antidepressant-md:ana-2bd47342d0de5cf7970950627aa6338cb55c174d")
PRIORITY = os.environ.get("ANA_PRIORITY", "medium")
CPU, MEMORY_MB = 16, 16384
TAGS = ("gb_eps2", "gb_eps1", "gb_eps4", "pb_eps2")


def group_name(key):
    return "eralpha-mm-" + key.replace("_", "-")


def ids_for(key):
    return [key] if "_rep" in key else [f"{key}_rep{r}" for r in (1, 2, 3)]


def make_folders(keys, dry):
    """Create mmgbsa-salad/<id>/<tag>/ and salad_logs/ in one serial rclone call before any node starts: Drive
    allows two folders with the same name, and nodes starting together would otherwise race to create them."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "salad_logs").mkdir()
        for key in keys:
            for i in ids_for(key):
                for t in TAGS:
                    (Path(tmp) / i / t).mkdir(parents=True)
        S.rclone("copy", tmp, f"{S.remote()}/mmgbsa-salad", "--create-empty-src-dirs", "--transfers", "1",
                 "--checkers", "1", dry=dry)


def cmd_create(a):
    env = S.rclone_env()
    make_folders(a.keys, a.dry_run)
    for key in a.keys:
        body = {
            "name": group_name(key), "display_name": f"eralpha MMGBSA {key.replace('_', ' ')} CPU",   # [ ,-.0-9A-Za-z] only
            "replicas": 1, "autostart_policy": False, "restart_policy": "never",
            "container": {
                "image": a.image, "image_caching": True, "priority": PRIORITY,
                "resources": {"cpu": CPU, "memory": MEMORY_MB, "gpu_classes": [], "shm_size": 1024},   # MB, for OpenMPI
                "environment_variables": env | {"DRIVE_REMOTE": S.remote(), "MM_KEYS": key, "NP": str(CPU)},
            },
        }
        if a.dry_run:
            S.mask(body["container"]["environment_variables"])
        res = S.http("POST", S.salad_url(), a.dry_run, body)
        if res:
            print(f"created {res['name']} (stopped, {CPU} vCPU / {MEMORY_MB // 1024} GB, {PRIORITY}), image {a.image}")


def cmd_start(a):
    for key in a.keys:
        S.http("POST", S.salad_url(f"/{group_name(key)}/start"), a.dry_run, {})
        print(f"started {group_name(key)}")


def cmd_stop(a):
    for key in a.keys:
        S.http("POST", S.salad_url(f"/{group_name(key)}/stop"), a.dry_run, {})
        print(f"stopped {group_name(key)}")


def cmd_status(a):
    done = S.drive_files("mmgbsa-salad", "DONE")
    groups = [g for g in S.http("GET", S.salad_url(), False).get("items", []) if g["name"].startswith("eralpha-mm-")]
    for g in sorted(groups, key=lambda g: g["name"]):
        st = g.get("current_state", {})
        inst = S.http("GET", S.salad_url(f"/{g['name']}/instances"), False).get("instances", []) \
            if st.get("status") == "running" else []
        states = ", ".join(f"{i.get('state')}" for i in inst) or "-"
        ids = [i for k in g["container"]["environment_variables"].get("MM_KEYS", "").split() for i in ids_for(k)]
        n_done = sum(f"{i}/{t}/DONE" in done for i in ids for t in TAGS)
        print(f"{g['name']:28s} {st.get('status', '?'):9s} instances: {states:20s} "
              f"jobs DONE on Drive {n_done}/{len(ids) * len(TAGS)}")


def main():
    S.load_env()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("create", "start", "stop"):
        s = sub.add_parser(c)
        s.add_argument("keys", nargs="+")
        if c == "create":
            s.add_argument("--image", default=IMAGE)
    sub.add_parser("status")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
