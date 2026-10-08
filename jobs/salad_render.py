#!/usr/bin/env python3
"""Render the trajectory animations (analysis/anim/render.py) on SaladCloud GPU nodes.

    python jobs/salad_render.py create NAME KEY [KEY ...]   # one stopped group rendering these systems in turn
    python jobs/salad_render.py start NAME [NAME ...]
    python jobs/salad_render.py status                      # groups, instances, MP4s on Drive anim-out/
    python jobs/salad_render.py stop NAME [NAME ...]
    python jobs/salad_render.py reallocate NAME [NAME ...] # move each group's instance to another node (slow node)
    python jobs/salad_render.py update NAME KEY [KEY ...] [--image ...] [--samples N] [--max-intro-s S]   # a stopped group, in place

Each group runs the render image (docker/render/Dockerfile) with ANIM_KEYS; its run_render.sh fetches the inputs that
mdana.anim wrote (uploaded to Drive anim-inputs/<key>/), renders with Cycles on the GPU (OptiX) and uploads
anim-out/<key>.mp4. Systems already on Drive are skipped, so a preempted and reallocated node resumes. The GPU class
is chosen as for the MD runs (GPU_CLASSES, default RTX 4070 Ti) at PRIORITY (default medium); the container exits
when its systems are done (restart_policy never) and `status` says which groups can be stopped.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit as S  # noqa: E402

IMAGE = os.environ.get("RENDER_IMAGE", "ghcr.io/masoudrezaei082-afk/eralpha-antidepressant-md:render")
CPU, MEMORY_MB = 4, 12288


def group_name(name):
    return "eralpha-render-" + name.replace("_", "-")


def group_env(a):
    return S.rclone_env() | {"DRIVE_REMOTE": S.remote(), "ANIM_KEYS": " ".join(a.keys), "SAMPLES": str(a.samples),
                             "MAX_INTRO_S": str(a.max_intro_s)}


def cmd_create(a):
    env = group_env(a)
    _, org, _ = S.need("SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT")
    classes = S.http("GET", f"{S.SALAD}/organizations/{org}/gpu-classes", False)["items"]
    chosen = [c for c in classes if any(c["name"].startswith(n) for n in S.GPU_NAMES)]
    if not chosen:
        sys.exit(f"no GPU class named like {S.GPU_NAMES}")
    for c in chosen:
        price = next((p["price"] for p in c["prices"] if p["priority"] == S.PRIORITY), "?")
        print(f"GPU class {c['name']}: ${price}/h at {S.PRIORITY} priority")
    body = {
        "name": group_name(a.name), "display_name": f"eralpha render {a.name.replace('_', ' ')} GPU",
        "replicas": 1, "autostart_policy": False, "restart_policy": "never",
        "container": {
            "image": a.image, "image_caching": True, "priority": S.PRIORITY,
            "resources": {"cpu": CPU, "memory": MEMORY_MB, "gpu_classes": [c["id"] for c in chosen]},
            "environment_variables": env,
        },
    }
    if a.dry_run:
        S.mask(body["container"]["environment_variables"])
    res = S.http("POST", S.salad_url(), a.dry_run, body)
    if res:
        print(f"created {res['name']} (stopped) for {' '.join(a.keys)}, image {a.image}")


def cmd_update(a):
    # Salad replaces environment_variables as a whole on PATCH (it is not merged key by key), so send all of them
    body = {"container": {"image": a.image, "environment_variables": group_env(a)}}
    if a.dry_run:
        S.mask(body["container"]["environment_variables"])
    S.http("PATCH", S.salad_url(f"/{group_name(a.name)}"), a.dry_run, body)
    print(f"updated {group_name(a.name)}: {' '.join(a.keys)}, image {a.image}, {a.samples} samples "
          "(Salad pulls the new version before the group can start)")


def cmd_start(a):
    for n in a.names:
        S.http("POST", S.salad_url(f"/{group_name(n)}/start"), a.dry_run, {})
        print(f"started {group_name(n)}")


def cmd_stop(a):
    for n in a.names:
        S.http("POST", S.salad_url(f"/{group_name(n)}/stop"), a.dry_run, {})
        print(f"stopped {group_name(n)}")


def cmd_reallocate(a):
    for n in a.names:
        inst = S.http("GET", S.salad_url(f"/{group_name(n)}/instances"), False).get("instances", [])
        for i in inst:
            iid = i.get("instance_id") or i.get("id")
            S.http("POST", S.salad_url(f"/{group_name(n)}/instances/{iid}/reallocate"), a.dry_run, {})
            print(f"reallocated {group_name(n)} instance {iid}")


def cmd_status(a):
    done = set(S.drive_files("anim-out", "*.mp4"))
    groups = [g for g in S.http("GET", S.salad_url(), False).get("items", [])
              if g["name"].startswith("eralpha-render-")]
    for g in sorted(groups, key=lambda g: g["name"]):
        st = g.get("current_state", {})
        inst = S.http("GET", S.salad_url(f"/{g['name']}/instances"), False).get("instances", []) \
            if st.get("status") == "running" else []
        states = ", ".join(f"{i.get('state')}" for i in inst) or "-"
        keys = g["container"]["environment_variables"].get("ANIM_KEYS", "").split()
        n = sum(f"{k}.mp4" in done for k in keys)
        print(f"{g['name']:28s} {st.get('status', '?'):9s} instances: {states:20s} MP4 on Drive {n}/{len(keys)}")


def main():
    S.load_env()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("create", "update"):
        c = sub.add_parser(cmd)
        c.add_argument("name")
        c.add_argument("keys", nargs="+")
        c.add_argument("--image", default=IMAGE)
        c.add_argument("--samples", type=int, default=48)
        c.add_argument("--max-intro-s", type=float, default=15,
                       help="give up on a node whose overview frames average more than this (s)")
    for cmd in ("start", "stop", "reallocate"):
        sub.add_parser(cmd).add_argument("names", nargs="+")
    sub.add_parser("status")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
