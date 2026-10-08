"""Stage 4: MM/GBSA (ε_in = 1, 2, 4) and MM/PBSA (ε_in = 2) end-point energies with gmx_MMPBSA.

One job = one replica × one method × one ε_in. Inputs come from the committed templates analysis/mmpbsa_gb.in and
mmpbsa_pb.in (placeholders @SYSTEM@ @REP@ @INTDIEL@). The processed trajectory keeps every 10 ps frame, so the
template frame numbers apply unchanged: frames 2001-10000 (20-100 ns), GB every 50th (160), PB every 150th (54).
Per-residue decomposition is kept only where `decomp` is set (ε_in = 2 by default).

Jobs run one after the other in a background process (`python -m mdana.mmgbsa run ...`); each finished job is
copied to Drive analysis/mmgbsa/<id>/<tag>/ with a DONE marker, and progress.json there is updated after every job,
so a disconnected runtime resumes where it stopped.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout as L

TEMPLATES = {"gb": L.REPO / "analysis" / "mmpbsa_gb.in", "pb": L.REPO / "analysis" / "mmpbsa_pb.in"}
EXPECTED_FRAMES = {"gb": 160, "pb": 54}
FIRST, LAST = 2001, 10000
ROOT = L.OUT / "mmgbsa"
SALAD_ROOT = L.DRIVE / "mmgbsa-salad"


def tag(method, eps):
    return f"{method}_eps{eps}"


def default_jobs(keys=None, decomp_eps=(2,)):
    """GB ε2 for every replica first (the main result), then GB ε1 and ε4, then PB ε2."""
    reps = [(s, r) for s, r in L.replicas(keys) if s.ligand]
    jobs = [(s, r, "gb", 2) for s, r in reps]
    jobs += [(s, r, "gb", e) for e in (1, 4) for s, r in reps]
    jobs += [(s, r, "pb", 2) for s, r in reps]
    return [dict(sys=s.key, rep=r, method=m, eps=e, decomp=(m == "gb" and e in decomp_eps)) for s, r, m, e in jobs]


def out_dir(job):
    return ROOT / L.BY_KEY[job["sys"]].rep_id(job["rep"]) / tag(job["method"], job["eps"])


def write_input(job, path):
    t = TEMPLATES[job["method"]].read_text()
    t = t.replace("@SYSTEM@", job["sys"]).replace("@REP@", str(job["rep"])).replace("@INTDIEL@", str(job["eps"]))
    if not job.get("decomp"):
        t = re.sub(r"&decomp.*?\n/\n", "", t, flags=re.S)
    # gmx_MMPBSA namelists do not accept trailing comments on every version: strip them
    lines = [re.sub(r"\s+#.*$", "", l) if not l.lstrip().startswith("#") else l for l in t.splitlines()]
    nums = {k: int(re.search(rf"{k}\s*=\s*(\d+)", t).group(1)) for k in ("startframe", "endframe", "interval")}
    n = len(range(nums["startframe"], nums["endframe"] + 1, nums["interval"]))
    assert (nums["startframe"], nums["endframe"]) == (FIRST, LAST) and n == EXPECTED_FRAMES[job["method"]], nums
    Path(path).write_text("\n".join(lines) + "\n")


def run_job(job, nproc=2, log=print):
    s = L.BY_KEY[job["sys"]]
    src = L.proc_dir(s, job["rep"])
    dst = out_dir(job)
    # a job is complete when DONE is there and, if it asks for the decomposition, so is the decomposition table;
    # a job finished on Salad (mmgbsa-salad/, which collect() prefers) counts too, or Colab would redo it
    for d in (SALAD_ROOT / dst.relative_to(ROOT), dst):
        if (d / "DONE").exists() and (not job.get("decomp") or (d / "FINAL_DECOMP_MMPBSA.csv").exists()):
            return "skip"
    if not (src / "processed.done").exists():
        raise RuntimeError(f"{s.rep_id(job['rep'])}: trajectory not processed yet")
    wd = L.WORK / "mmgbsa" / s.rep_id(job["rep"]) / tag(job["method"], job["eps"])
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    cache = L.ensure(L.WORK / "mmgbsa" / s.rep_id(job["rep"]) / "inputs")
    for f in ("processed.xtc", "complex.tpr", "complex.ndx", "topol_complex.top"):
        if not (cache / f).exists() or (cache / f).stat().st_size != (src / f).stat().st_size:
            shutil.copyfile(src / f, cache / f)
        os.symlink(cache / f, wd / f)
    write_input(job, wd / "mmpbsa.in")
    cmd = ["gmx_MMPBSA", "-O", "-i", "mmpbsa.in", "-cs", "complex.tpr", "-ci", "complex.ndx", "-cg", "0", "1",
           "-ct", "processed.xtc", "-cp", "topol_complex.top", "-o", "FINAL_RESULTS_MMPBSA.dat",
           "-eo", "FINAL_RESULTS_MMPBSA.csv", "-nogui"]
    if job.get("decomp"):
        cmd += ["-do", "FINAL_DECOMP_MMPBSA.dat", "-deo", "FINAL_DECOMP_MMPBSA.csv"]
    if nproc > 1:
        cmd = ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", str(nproc), cmd[0], "MPI", *cmd[1:]]
    t0 = time.time()
    with open(wd / "run.log", "w") as fh:
        r = subprocess.run(cmd, cwd=wd, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if r.returncode or not (wd / "FINAL_RESULTS_MMPBSA.dat").exists():
        tail = (wd / "run.log").read_text()[-3000:]
        raise RuntimeError(f"gmx_MMPBSA failed for {dst.name} of {s.rep_id(job['rep'])}:\n{tail}")
    return finish(job, wd, wall, log)


def finish(job, wd, wall, log=print):
    """Parse and check a finished gmx_MMPBSA run directory, copy its outputs to Drive and mark the job DONE."""
    s, dst = L.BY_KEY[job["sys"]], out_dir(job)
    res = parse_dat(wd / "FINAL_RESULTS_MMPBSA.dat")
    if res["n_frames"] != EXPECTED_FRAMES[job["method"]]:
        raise RuntimeError(f"{res['n_frames']} frames analysed, expected {EXPECTED_FRAMES[job['method']]}")
    res.update(job=job, wall_s=round(wall, 1))
    L.ensure(dst)
    for f in ("FINAL_RESULTS_MMPBSA.dat", "FINAL_RESULTS_MMPBSA.csv", "FINAL_DECOMP_MMPBSA.dat",
              "FINAL_DECOMP_MMPBSA.csv", "mmpbsa.in", "run.log", "gmx_MMPBSA.log"):
        if (wd / f).exists():
            shutil.copyfile(wd / f, dst / f)
    (dst / "result.json").write_text(json.dumps(res, indent=1))
    (dst / "DONE").write_text("ok\n")
    shutil.rmtree(wd, ignore_errors=True)
    log(f"{s.rep_id(job['rep'])} {dst.name}: ΔTOTAL {res['delta']['TOTAL']['mean']:.2f} kcal/mol, "
        f"{res['n_frames']} frames, {wall / 60:.1f} min")
    return res


# ------------------------------------------------------------------------------------------------ parsing
def parse_dat(path):
    """Delta (complex − receptor − ligand) terms from FINAL_RESULTS_MMPBSA.dat (the last method section)."""
    text = Path(path).read_text()
    m = re.search(r"Calculations performed using (\d+) complex frames", text)
    n_frames = int(m.group(1)) if m else None
    blocks = re.split(r"Delta \(Complex - Receptor - Ligand\):", text)
    if len(blocks) < 2:
        raise RuntimeError(f"no Delta block in {path}")
    delta = {}
    cols = ("mean", "sd_prop", "sd", "sem_prop", "sem", "block_sd", "block_sem")   # 1.7.0 adds the block columns
    for line in blocks[-1].splitlines():
        m = re.match(r"Δ(\S+(?: \S+)?)\s+((?:-?\d+\.\d+\s*){5,7})$", line.strip())
        if m:
            vals = [float(x) for x in m.group(2).split()]
            delta[m.group(1).replace(" ", "_")] = dict(zip(cols, vals))
        if line.startswith("----") and delta and "TOTAL" in delta:
            break
    if "TOTAL" not in delta:
        raise RuntimeError(f"ΔTOTAL not found in {path}")
    return {"n_frames": n_frames, "delta": delta}


def parse_frames(path):
    """Per-frame Delta energies from the -eo CSV (columns: Frame #, ..., TOTAL)."""
    lines = Path(path).read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("Delta Energy Terms"))
    except StopIteration:
        raise RuntimeError(f"no Delta Energy Terms in {path}")
    header = lines[start + 1].split(",")
    rows = []
    for l in lines[start + 2:]:
        f = l.split(",")
        if not l.strip() or not f[0].strip().isdigit():
            break
        rows.append([float(x) for x in f])
    df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    return df.rename(columns={"Frame #": "frame"})


def parse_decomp(path):
    """Per-residue total contribution (mean over frames) from the DELTAS block of the -deo CSV."""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("DELTAS"))
    j = next(k for k in range(i, len(lines)) if lines[k].strip().startswith("Total Decomposition Contribution"))
    header = [h.strip() for h in lines[j + 1].split(",")]
    rows = []
    for l in lines[j + 2:]:
        f = [x.strip() for x in l.split(",")]
        if len(f) != len(header) or not f[0].isdigit():
            break
        rows.append(f)
    df = pd.DataFrame(rows, columns=header)
    num = [c for c in header if c not in ("Frame #", "Residue")]
    df[num] = df[num].astype(float)
    g = df.groupby("Residue", sort=False)[num].mean().reset_index()
    parts = g.Residue.str.split(":", expand=True)            # e.g. R:A:GLU:47 / L:B:LIG:256
    g["side"], g["resname"], g["resnum"] = parts[0], parts[2], parts[3].astype(int)
    return g


def collect():
    """All finished jobs -> one long table (per replica) and per-frame / decomposition tables."""
    rows, frames, decomp, seen = [], [], [], set()
    # Salad results first (written by rclone to <DRIVE>/mmgbsa-salad, see jobs/salad_mmgbsa.py), then the Colab ones;
    # a job present in both is taken once, from Salad (its gb_eps2 carries the decomposition)
    files = sorted(SALAD_ROOT.glob("*/*/result.json")) + sorted(ROOT.glob("*/*/result.json"))
    for res_file in files:
        r = json.loads(res_file.read_text())
        j = r["job"]
        if (j["sys"], j["rep"], j["method"], j["eps"]) in seen:
            continue
        seen.add((j["sys"], j["rep"], j["method"], j["eps"]))
        base = dict(sys=j["sys"], rep=j["rep"], method=j["method"], eps=j["eps"], n_frames=r["n_frames"],
                    wall_s=r.get("wall_s"))
        rows.append({**base, **{f"d_{k}": v["mean"] for k, v in r["delta"].items()},
                     "frame_sem": r["delta"]["TOTAL"]["sem"]})
        csv = res_file.parent / "FINAL_RESULTS_MMPBSA.csv"
        if csv.exists():
            f = parse_frames(csv)
            frames.append(f.assign(**{k: base[k] for k in ("sys", "rep", "method", "eps")}))
        dcsv = res_file.parent / "FINAL_DECOMP_MMPBSA.csv"
        if dcsv.exists():
            decomp.append(parse_decomp(dcsv).assign(**{k: base[k] for k in ("sys", "rep", "method", "eps")}))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return pd.DataFrame(rows), cat(frames), cat(decomp)


# ------------------------------------------------------------------------------------------------ old results
def parse_old(path):
    """ΔTOTAL of the original single-run gmx_MMPBSA 1.6.4 outputs (first block GB, second PB)."""
    vals = [float(l.split()[1]) for l in Path(path).read_text().splitlines() if l.startswith("ΔTOTAL")]
    out = {"gb": vals[0] if vals else np.nan, "pb": vals[1] if len(vals) > 1 else np.nan}
    m = re.search(r"Calculations performed using (\d+) complex frames", Path(path).read_text())
    out["n_frames"] = int(m.group(1)) if m else None
    return out


# ------------------------------------------------------------------------------------------------ runner
def progress_write(state):
    L.ensure(ROOT)
    tmp = ROOT / "progress.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(ROOT / "progress.json")


def run_queue(jobs, nproc=2):
    state = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(jobs), "done": 0, "failed": [],
             "current": None, "pid": os.getpid(), "wall_s": {}}
    for k, job in enumerate(jobs):
        name = f"{L.BY_KEY[job['sys']].rep_id(job['rep'])}/{tag(job['method'], job['eps'])}"
        state["current"] = name
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        progress_write(state)
        try:
            r = run_job(job, nproc=nproc, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
            if isinstance(r, dict):
                state["wall_s"][name] = r["wall_s"]
            state["done"] += 1
        except Exception as e:                       # keep going; a failed job is reported and can be rerun
            print(time.strftime("%H:%M:%S"), "FAILED", name, str(e)[-1500:], flush=True)
            state["failed"].append({"job": name, "error": str(e)[-800:]})
    state["current"] = None
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_write(state)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mdana.mmgbsa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run jobs in this process (start it with nohup)")
    r.add_argument("--keys", nargs="*", help="system keys or key_repN ids (default: all liganded)")
    r.add_argument("--only", nargs="*", help="tags to run, e.g. gb_eps2 pb_eps2")
    r.add_argument("--decomp-eps", nargs="*", type=int, default=[2])
    r.add_argument("--np", type=int, default=2)
    sub.add_parser("status", help="print progress.json")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        p = ROOT / "progress.json"
        print(p.read_text() if p.exists() else "no progress.json yet")
        return
    jobs = default_jobs(a.keys, tuple(a.decomp_eps))
    if a.only:
        jobs = [j for j in jobs if tag(j["method"], j["eps"]) in a.only]
    run_queue(jobs, nproc=a.np)


if __name__ == "__main__":
    sys.exit(main())
