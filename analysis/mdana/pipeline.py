"""Background driver for the structural stages: python -m mdana.pipeline structure [--keys ...]

Estradiol rep1 goes first (the apo receptor uses its pocket). Progress is appended to
<analysis>/structure_progress.json after every replica; a failed replica is reported and skipped.
"""
import argparse
import json
import sys
import time
import traceback

from . import layout as L
from . import metrics as M
from . import traj as TR


def structure(keys=None, force=False):
    reps = L.replicas(keys)
    reps.sort(key=lambda sr: (sr[0].key != L.REFERENCE or sr[1] != 1, not sr[0].ligand))
    prog = L.OUT / "structure_progress.json"
    state = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(reps), "done": [], "failed": []}
    for s, r in reps:
        state["current"] = s.rep_id(r)
        L.ensure(L.OUT)
        prog.write_text(json.dumps(state, indent=1))
        t0 = time.time()
        try:
            checks = TR.process(s, r, force=force)
            M.run(s, r, force=force)
            state["done"].append({"id": s.rep_id(r), "min": round((time.time() - t0) / 60, 1),
                                  "lig_mindist_max": checks.get("max_min_dist_ligand_protein_A"),
                                  "src2_mindist_max": checks.get("max_min_dist_src2_receptor_A")})
            print(time.strftime("%H:%M:%S"), "done", s.rep_id(r), f"{(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(time.strftime("%H:%M:%S"), "FAILED", s.rep_id(r), err, flush=True)
            state["failed"].append({"id": s.rep_id(r), "error": err[-1500:]})
    state["current"] = None
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    prog.write_text(json.dumps(state, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mdana.pipeline")
    ap.add_argument("stage", choices=["structure"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    structure(a.keys, a.force)


if __name__ == "__main__":
    sys.exit(main())
