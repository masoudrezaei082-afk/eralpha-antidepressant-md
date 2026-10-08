"""Animation inputs: one representative replica per ligand, reduced to a short smoothed trajectory plus per-frame
ligand–protein interactions, for rendering on a GPU node (analysis/anim/render.py).

The representative replica is the one whose 20–100 ns mean ligand RMSD is closest to the median of the three
replica means. 0–100 ns is compressed to N_OUT frames; each output frame is the mean of 2*HALF_WIN+1 consecutive
raw frames (10 ps apart) of the pocket-fitted trajectory, which removes thermal jitter without hiding real motion.
Interactions are judged on the raw frames: an H-bond or salt bridge is drawn if it is present in at least
MIN_PRESENT of the raw frames in the window, with the same criteria as metrics.py.
"""
import json
import shutil

import numpy as np
import pandas as pd

from . import layout as L
from . import metrics as M
from . import traj as TR

N_OUT = 600                 # 20 s at 30 fps
HALF_WIN = 3                # ±30 ps smoothing window
MIN_PRESENT = 3             # of the 7 raw frames
MAX_SHOWN = 12              # pocket residues drawn as sticks
SHOW_OCC = 0.30             # contact occupancy for a residue to be drawn


def anim_dir(s):
    return L.OUT / "anim" / s.key


def representative(s):
    means = {}
    for rep in L.REPLICAS:
        ts = pd.read_parquet(L.proc_dir(s, rep) / "ts.parquet")
        means[rep] = float(ts[ts.time_ns >= L.T_EQ_NS].lig_rmsd.mean())
    med = float(np.median(list(means.values())))
    rep = min(means, key=lambda r: (abs(means[r] - med), r))
    return rep, means


def hbond_atoms(d_pairs, acc, side_of, pos):
    """Ligand–protein H-bonds in one frame as (donor, hydrogen, acceptor) atom indices."""
    from MDAnalysis.lib.distances import capped_distance
    out = set()
    if not len(d_pairs) or not len(acc):
        return out
    pairs = capped_distance(pos[d_pairs[:, 0]], pos[acc], M.HB_DIST, return_distances=False)
    for di, ai in pairs:
        d, h = d_pairs[di]
        a = acc[ai]
        if side_of[d] == side_of[a]:
            continue
        v1, v2 = pos[d] - pos[h], pos[a] - pos[h]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) >= M.HB_ANGLE:
            out.add((int(d), int(h), int(a)))
    return out


def open_direction(lig_cog, prot_xyz, n_dir=400, radius=5.0, depth=30.0):
    """Unit vector from the ligand out of the protein with the fewest protein heavy atoms in a 5 Å cylinder."""
    k = np.arange(n_dir) + 0.5
    phi, theta = np.arccos(1 - 2 * k / n_dir), np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    rel = prot_xyz - lig_cog
    best, best_n = None, None
    for d in dirs:
        t = rel @ d
        perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
        n = int(((t > 0) & (t < depth) & (perp < radius)).sum())
        if best_n is None or n < best_n:
            best, best_n = d, n
    return best, best_n


def prep(s, force=False):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from MDAnalysis.lib.distances import capped_distance

    out = L.ensure(anim_dir(s))
    if (out / "anim.done").exists() and not force:
        return json.loads((out / "meta.json").read_text())
    rep, means = representative(s)
    work = M.local_copy(s, rep)
    u = TR.universe(work, work / "processed.xtc")
    n_raw = u.trajectory.n_frames
    centres = np.round(np.linspace(HALF_WIN, n_raw - 1 - HALF_WIN, N_OUT)).astype(int)

    rec, pep = f"resid {L.REC_RESID[0]}:{L.REC_RESID[1]}", f"resid {L.PEP_RESID[0]}:{L.PEP_RESID[1]}"
    prot = u.select_atoms(f"({rec} or {pep})")
    lig = u.select_atoms(f"resname {L.LIGAND_RESNAME}")
    d_pairs, acc = M.donors_acceptors(prot + lig)
    side_of = np.zeros(u.atoms.n_atoms, dtype=int)
    side_of[prot.ix] = 1
    cationic = bool(lig.charges.sum() > 0.5)
    lig_np = lig.select_atoms("element N")
    lig_np = lig_np[[any(n.element == "H" for n in a.bonded_atoms) for a in lig_np]] if cationic else lig_np[:0]
    acidic = prot.select_atoms("(resname ASP ASH and name OD1 OD2) or (resname GLU GLH and name OE1 OE2)")

    coords = np.zeros((N_OUT, u.atoms.n_atoms, 3), dtype=np.float32)
    frames = []
    for k, c in enumerate(centres):
        hb_count, sb_count = {}, {}
        for i in range(c - HALF_WIN, c + HALF_WIN + 1):
            u.trajectory[i]
            pos = u.atoms.positions
            coords[k] += pos
            for t in hbond_atoms(d_pairs, acc, side_of, pos):
                hb_count[t] = hb_count.get(t, 0) + 1
            if lig_np.n_atoms:
                for li, ai in capped_distance(lig_np.positions, acidic.positions, M.SALT_CUT, return_distances=False):
                    key = (int(lig_np[li].ix), int(acidic[ai].ix))
                    sb_count[key] = sb_count.get(key, 0) + 1
        coords[k] /= 2 * HALF_WIN + 1
        hb = sorted(t for t, n in hb_count.items() if n >= MIN_PRESENT)
        # one salt-bridge line per (ligand N, residue): the carboxylate O seen most often in the window
        best = {}
        for (li, ai), n in sb_count.items():
            if n < MIN_PRESENT:
                continue
            key = (li, int(u.atoms[ai].resindex))
            if key not in best or n > best[key][1]:
                best[key] = (ai, n)
        sb = sorted((li, ai) for (li, _), (ai, _) in best.items())
        u.trajectory[c]
        frames.append({"raw_frame": int(c), "time_ns": round(float(u.trajectory.time) / 1000.0, 3),
                       "hbonds": [list(t) for t in hb], "salt": [list(p) for p in sb]})

    ts = pd.read_parquet(L.proc_dir(s, rep) / "ts.parquet")
    for f in frames:
        f["lig_rmsd"] = round(float(ts.lig_rmsd.iloc[f["raw_frame"]]), 3)

    # residues drawn as sticks: frequent contacts in this replica, plus the estradiol anchors
    occ = pd.read_csv(L.proc_dir(s, rep) / "occupancy.csv")
    anchor_ids = [L.gro_resid(x) for x in L.ANCHORS.values()]
    keep = occ[(occ.contact >= SHOW_OCC) & ~occ.resid.isin(anchor_ids)].sort_values("contact", ascending=False)
    shown = anchor_ids + [int(r) for r in keep.resid[:MAX_SHOWN - len(anchor_ids)]]
    resinfo = []
    for r in shown:
        res = u.select_atoms(f"resid {r}").residues[0]
        resinfo.append({"resid": int(r), "label": L.label_resid(int(r), res.resname),
                        "anchor": int(r) in anchor_ids,
                        "contact_occ": float(occ.loc[occ.resid == r, "contact"].max()) if (occ.resid == r).any() else 0.0})

    # per-frame interaction flags for the residues that make H-bonds or salt bridges anywhere in the clip
    resindex_of = {int(a.ix): int(a.resid) for a in prot}
    partners = set()
    for f in frames:
        for d, h, a in f["hbonds"]:
            partners.add(resindex_of.get(a, resindex_of.get(d)))
        for li, ai in f["salt"]:
            partners.add(resindex_of[ai])
    partners.discard(None)
    track = sorted(set(anchor_ids) | partners, key=lambda r: (r not in anchor_ids, r))
    for f in frames:
        hit = {resindex_of.get(a, resindex_of.get(d)) for d, h, a in f["hbonds"]} | \
              {resindex_of[ai] for _, ai in f["salt"]}
        f["contacts"] = [int(r) for r in track if r in hit]

    lig_cog0 = coords[0][lig.ix].mean(0)
    heavy = prot.select_atoms("not name H*")
    view, n_block = open_direction(lig_cog0, coords[0][heavy.ix])

    um = mda.Merge(u.atoms)
    um.load_new(coords, format=MemoryReader, dt=float(np.diff(centres).mean() * L.DT_PS))
    # separate chains, so the cartoon does not bridge receptor and SRC2
    chain = np.full(um.atoms.n_atoms, "X", dtype=object)
    chain[prot.select_atoms(rec).ix], chain[prot.select_atoms(pep).ix], chain[lig.ix] = "A", "B", "L"
    um.add_TopologyAttr("chainIDs", list(chain))
    um.atoms.write(str(out / "complex.pdb"), bonds="all")
    with mda.Writer(str(out / "anim.xtc"), um.atoms.n_atoms) as w:
        for _ in um.trajectory:
            w.write(um.atoms)

    meta = {
        "key": s.key, "label": s.label, "color": s.color, "replica": rep,
        "replica_mean_lig_rmsd": {str(k): round(v, 3) for k, v in means.items()},
        "n_frames": N_OUT, "half_window_frames": HALF_WIN, "raw_dt_ps": L.DT_PS,
        "n_atoms": int(u.atoms.n_atoms),
        "ligand_ix": [int(i) for i in lig.ix],
        "receptor_resid": list(L.REC_RESID), "peptide_resid": list(L.PEP_RESID),
        "h12_resid": [L.gro_resid(x) for x in L.H12_CRYSTAL],
        "shown_residues": resinfo,
        "tracked_residues": [{"resid": int(r), "label": L.label_resid(int(r), u.select_atoms(f"resid {r}").residues[0].resname),
                              "anchor": int(r) in anchor_ids} for r in track],
        "cationic": cationic,
        "view_dir": [float(x) for x in view], "view_blocking_atoms": n_block,
        "ligand_cog0": [float(x) for x in lig_cog0],
        "criteria": {"hbond": f"D–A ≤ {M.HB_DIST} Å, D–H–A ≥ {M.HB_ANGLE:g}°",
                     "salt": f"N⁺–O⁻ ≤ {M.SALT_CUT} Å",
                     "present": f"in ≥ {MIN_PRESENT} of {2 * HALF_WIN + 1} raw frames"},
    }
    (out / "frames.json").write_text(json.dumps(frames))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "anim.done").write_text("ok\n")
    return meta


def prep_all(force=False):
    res = {}
    for s in L.LIGANDED:
        res[s.key] = prep(s, force=force)
        print(s.key, "replica", res[s.key]["replica"], "blocking atoms", res[s.key]["view_blocking_atoms"], flush=True)
    return res


def bundle(dst):
    """Copy the small animation inputs to a local folder (to be moved to the render node's storage)."""
    for s in L.LIGANDED:
        d = L.ensure(dst / s.key)
        for f in ("complex.pdb", "anim.xtc", "frames.json", "meta.json"):
            shutil.copyfile(anim_dir(s) / f, d / f)
