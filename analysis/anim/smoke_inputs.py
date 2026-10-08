#!/usr/bin/env python3
"""Synthetic inputs for the render smoke test (CI has no trajectory data): an ideal α-helix "receptor" (resid 1–24),
a short helical "peptide" (resid 26–30) and a phenol-like ligand, in the format mdana.anim writes.

    python smoke_inputs.py OUT_DIR
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def helix(n, start, origin, axis_shift):
    """Backbone N, CA, C, O + CB of an ideal α-helix (100° and 1.5 Å per residue)."""
    atoms = []
    for i in range(n):
        th = math.radians(100 * i)
        z = 1.5 * i
        for name, r, dth, dz, el in (("N", 1.55, -28, -0.9, "N"), ("CA", 2.3, 0, 0, "C"), ("C", 1.6, 28, 0.9, "C"),
                                     ("O", 1.9, 40, 2.0, "O"), ("CB", 3.3, 5, -0.5, "C")):
            a = th + math.radians(dth)
            atoms.append((start + i, name, el, origin + np.array([r * math.cos(a) + axis_shift, r * math.sin(a), z + dz])))
    return atoms


def main(out):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = helix(24, 1, np.zeros(3), 0.0)
    pep = helix(5, 26, np.array([0.0, 0.0, 12.0]), 11.0)
    ring = [(31, f"C{i + 1}", "C", np.array([6.0 + 1.4 * math.cos(math.radians(60 * i)),
                                              1.4 * math.sin(math.radians(60 * i)), 18.0])) for i in range(6)]
    lig = ring + [(31, "O1", "O", np.array([8.8, 0.0, 18.0])), (31, "H1", "H", np.array([9.4, 0.7, 18.0]))]
    atoms = rec + pep + lig
    n = len(atoms)
    resids = [a[0] for a in atoms]
    res_list = sorted(set(resids))
    u = mda.Universe.empty(n, n_residues=len(res_list), atom_resindex=[res_list.index(r) for r in resids],
                           trajectory=True)
    u.add_TopologyAttr("names", [a[1] for a in atoms])
    u.add_TopologyAttr("elements", [a[2] for a in atoms])
    u.add_TopologyAttr("types", [a[2] for a in atoms])
    u.add_TopologyAttr("resids", res_list)
    u.add_TopologyAttr("resnames", ["LIG" if r == 31 else "ALA" for r in res_list])
    u.add_TopologyAttr("chainIDs", ["L" if r == 31 else "B" if r >= 26 else "A" for r in resids])
    u.add_TopologyAttr("segids", ["SYS"])
    bonds = []
    idx = {(a[0], a[1]): i for i, a in enumerate(atoms)}
    for r in res_list[:-1]:
        for b in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
            bonds.append((idx[(r, b[0])], idx[(r, b[1])]))
        if (r + 1, "N") in idx:
            bonds.append((idx[(r, "C")], idx[(r + 1, "N")]))
    for i in range(6):
        bonds.append((idx[(31, f"C{i + 1}")], idx[(31, f"C{(i + 1) % 6 + 1}")]))
    bonds += [(idx[(31, "C1")], idx[(31, "O1")]), (idx[(31, "O1")], idx[(31, "H1")])]
    u.add_TopologyAttr("bonds", bonds)
    xyz0 = np.array([a[3] for a in atoms], dtype=np.float32)
    nf = 120
    rng = np.random.default_rng(0)
    traj = np.stack([xyz0 + rng.normal(0, 0.15, xyz0.shape).astype(np.float32) for _ in range(nf)])
    traj[:, idx[(31, "C1")]:, 0] += np.linspace(0, 2.0, nf)[:, None]
    u.load_new(traj, format=MemoryReader, dt=166.7)
    u.dimensions = [80, 80, 80, 90, 90, 90]
    u.atoms.write(str(out / "complex.pdb"), bonds="all")
    with mda.Writer(str(out / "anim.xtc"), n) as w:
        for _ in u.trajectory:
            w.write(u.atoms)

    o_acc = idx[(10, "O")]
    frames = [{"raw_frame": k, "time_ns": round(100 * k / (nf - 1), 3), "lig_rmsd": round(0.5 + k / nf, 3),
               "hbonds": [[idx[(31, "O1")], idx[(31, "H1")], o_acc]] if k % 3 else [],
               "salt": [[idx[(31, "O1")], idx[(12, "O")]]] if k % 5 == 0 else [],
               "contacts": [10] if k % 3 else []} for k in range(nf)]
    meta = {"key": "smoke", "label": "Smoke test", "color": "#000000", "replica": 1,
            "replica_mean_lig_rmsd": {"1": 1.0, "2": 1.1, "3": 0.9}, "n_frames": nf, "half_window_frames": 3,
            "raw_dt_ps": 10.0, "n_atoms": n, "ligand_ix": [idx[k] for k in idx if k[0] == 31],
            "receptor_resid": [1, 24], "peptide_resid": [26, 30], "h12_resid": [20, 24],
            "shown_residues": [{"resid": r, "label": f"Ala{r}", "anchor": r == 10, "contact_occ": 0.5}
                               for r in (8, 10, 12)],
            "tracked_residues": [{"resid": 10, "label": "Ala10", "anchor": True},
                                 {"resid": 12, "label": "Ala12", "anchor": False}],
            "cationic": False, "view_dir": [1.0, 0.0, 0.0], "view_blocking_atoms": 0,
            "ligand_cog0": [float(x) for x in xyz0[-8:].mean(0)],
            "criteria": {"hbond": "D–A ≤ 3.5 Å, D–H–A ≥ 150°", "salt": "N–O ≤ 4.0 Å", "present": "in ≥ 3 of 7 raw frames"}}
    (out / "frames.json").write_text(json.dumps(frames))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("smoke inputs in", out)


if __name__ == "__main__":
    main(sys.argv[1])
