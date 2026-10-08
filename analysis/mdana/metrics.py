"""Stage 3: structural metrics of one replica, from the processed (whole, imaged, pocket-fitted) trajectory.

Per frame (10 ps):
  lig_rmsd        ligand heavy-atom RMSD to the starting (docked / crystal) pose, symmetry-corrected (spyrmsd), in the
                  pocket-Cα frame (no ligand refit)                                                          [Å]
  lig_rmsd_plain  the same without symmetry correction (a check: never smaller than lig_rmsd)
  lig_pocket_dist distance between the ligand centroid and the pocket centroid                               [Å]
  lig_disp        displacement of the ligand centroid from its starting position                             [Å]
  lig_min_dist    shortest ligand–protein heavy-atom distance (> ~4 Å means the ligand has left the pocket)   [Å]
  bb_rmsd         receptor backbone RMSD after a fit on the core (termini, modelled loop and helix 12 excluded) [Å]
  h12_rmsd        helix-12 Cα RMSD (Glu536–Leu544) on the core fit (no refit)                                [Å]
  src2_rmsd       SRC2 peptide Cα RMSD on the core fit (no refit)                                            [Å]
  clamp_k362      Lys362 NZ – SRC2 backbone O, shortest distance (charge clamp)                              [Å]
  clamp_e542      Glu542 carboxylate O – SRC2 backbone N, shortest distance (charge clamp)                   [Å]
Every 50 ps from 20 ns on (CONTACT_STRIDE): per residue occupancy of heavy-atom contact (< 4 Å), hydrogen bond
(D–A < 3.5 Å, D–H···A > 150°), polar contact (any ligand N/O
within 3.5 Å of a residue N/O, no angle criterion: also catches water-bridged or poorly aligned H-bonds) and salt bridge (ligand N–H⁺ to Asp/Glu carboxylate O < 4 Å).
Cα RMSF from 20 ns on, after the core fit. The QC of the thermodynamic state comes from the joined edr.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout as L
from . import traj as TR

HB_DIST, HB_ANGLE, CONTACT_CUT, SALT_CUT = 3.5, 150.0, 4.0, 4.0


def local_copy(s, rep):
    """Processed files on local disk (copied from Drive if this VM has not produced them)."""
    d, work = L.proc_dir(s, rep), L.ensure(L.WORK / s.rep_id(rep))
    for f in ("processed.xtc", "complex_ref.gro", "topol_complex.top"):
        if not (work / f).exists() or (work / f).stat().st_size != (d / f).stat().st_size:
            shutil.copyfile(d / f, work / f)
    return work


def heavy_graph(ag):
    """Atomic numbers and adjacency matrix of the heavy atoms of a molecule (for spyrmsd)."""
    Z = {"C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35, "I": 53, "P": 15}
    idx = {a.ix: k for k, a in enumerate(ag)}
    adj = np.zeros((ag.n_atoms, ag.n_atoms), dtype=int)
    for b in ag.bonds:
        i, j = b.atoms.ix
        if i in idx and j in idx:
            adj[idx[i], idx[j]] = adj[idx[j], idx[i]] = 1
    return np.array([Z[e] for e in ag.elements]), adj


def symm_rmsd(ref_xyz, frames_xyz, Z, adj):
    from spyrmsd import rmsd as srmsd
    return np.asarray(srmsd.symmrmsd(ref_xyz, list(frames_xyz), Z, Z, adj, adj,
                                     center=False, minimize=False, cache=True), dtype=float)


def donors_acceptors(ag):
    """(donor heavy index, hydrogen index) pairs and acceptor indices (O; N without H; F is not counted)."""
    pairs, acc = [], []
    for a in ag:
        if a.element not in ("N", "O"):
            continue
        hs = [n for n in a.bonded_atoms if n.element == "H"]
        for h in hs:
            pairs.append((a.ix, h.ix))
        if a.element == "O" or not hs:
            acc.append(a.ix)
    return np.array(pairs, dtype=int).reshape(-1, 2), np.array(acc, dtype=int)


def hbond_residues(u, d_pairs, acc, side_of, pos):
    """Protein residues (resindex) H-bonded to the ligand in this frame, either direction."""
    from MDAnalysis.lib.distances import capped_distance
    found = set()
    if not len(d_pairs) or not len(acc):
        return found
    donors_heavy = pos[d_pairs[:, 0]]
    pairs = capped_distance(donors_heavy, pos[acc], HB_DIST, return_distances=False)
    for di, ai in pairs:
        d, h = d_pairs[di]
        a = acc[ai]
        if side_of[d] == side_of[a]:                   # ligand–ligand or protein–protein
            continue
        v1, v2 = pos[d] - pos[h], pos[a] - pos[h]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) >= HB_ANGLE:
            found.add(u.atoms[a if side_of[a] == 1 else d].resindex)
    return found


def compute(s, rep):
    import MDAnalysis as mda
    from MDAnalysis.analysis.align import rotation_matrix
    from MDAnalysis.lib.distances import capped_distance, distance_array

    work = local_copy(s, rep)
    u = TR.universe(work, work / "processed.xtc")
    ref = TR.universe(work)
    checks = json.loads((L.proc_dir(s, rep) / "traj_checks.json").read_text())
    pocket = checks["pocket_resids"]

    rec, pep = f"resid {L.REC_RESID[0]}:{L.REC_RESID[1]}", f"resid {L.PEP_RESID[0]}:{L.PEP_RESID[1]}"
    loop = tuple(L.gro_resid(x) for x in L.LOOP_CRYSTAL)
    h12 = tuple(L.gro_resid(x) for x in L.H12_CRYSTAL)
    core_sel = (f"backbone and resid {L.REC_RESID[0] + 3}:{h12[0] - 1} "
                f"and not resid {loop[0]}:{loop[1]}")
    sels = {
        "core": core_sel,
        "ca": f"name CA and ({rec} or {pep})",
        "h12": f"name CA and resid {h12[0]}:{h12[1]}",
        "pep_ca": f"name CA and {pep}",
        "k362": f"resid {L.gro_resid(362)} and name NZ",
        "e542": f"resid {L.gro_resid(542)} and name OE1 OE2",
        "pep_o": f"{pep} and name O", "pep_n": f"{pep} and name N",
        "pocket_heavy": "not name H* and resid " + " ".join(map(str, pocket)),
    }
    m = {k: u.select_atoms(v) for k, v in sels.items()}
    r = {k: ref.select_atoms(v) for k, v in sels.items()}
    ref_core = r["core"].positions - r["core"].positions.mean(0)

    prot = u.select_atoms(f"({rec} or {pep})")
    prot_heavy = prot.select_atoms("not name H*")
    lig = u.select_atoms(f"resname {L.LIGAND_RESNAME}")
    lig_heavy = lig.select_atoms("not name H*")
    has_lig = lig.n_atoms > 0

    if has_lig:
        lig_ref = ref.select_atoms(f"resname {L.LIGAND_RESNAME} and not name H*").positions.copy()
        lig_ref_cog = lig_ref.mean(0)
        both = prot + lig
        d_pairs, acc = donors_acceptors(both)
        side_of = np.zeros(u.atoms.n_atoms, dtype=int)
        side_of[prot.ix] = 1                                         # 1 protein, 0 ligand
        acc_lig = [i for i in acc if side_of[i] == 0]
        cationic = bool(lig.charges.sum() > 0.5)
        lig_np = lig.select_atoms("element N").atoms
        lig_np = lig_np[[any(n.element == "H" for n in a.bonded_atoms) for a in lig_np]] if cationic else lig_np[:0]
        acidic = prot.select_atoms("(resname ASP ASH and name OD1 OD2) or (resname GLU GLH and name OE1 OE2)")
        lig_polar = lig.select_atoms("element N O")
        prot_polar = prot.select_atoms("element N O")

    n = u.trajectory.n_frames
    i_eq = int(round(L.T_EQ_NS * 1000 / L.DT_PS))
    cols = {k: np.full(n, np.nan) for k in ("lig_rmsd_plain", "lig_pocket_dist", "lig_disp", "lig_min_dist",
                                             "bb_rmsd", "h12_rmsd", "src2_rmsd", "clamp_k362", "clamp_e542")}
    time_ns = np.zeros(n)
    lig_xyz = np.zeros((n, lig_heavy.n_atoms, 3), dtype=np.float32) if has_lig else None
    ca_sum = np.zeros((m["ca"].n_atoms, 3))
    ca_sq = np.zeros(m["ca"].n_atoms)
    n_eq = 0
    n_res = len(u.residues)
    cnt = {k: np.zeros(n_res) for k in ("contact", "polar", "hbond", "saltbridge")}
    n_sampled = 0

    for i, ts in enumerate(u.trajectory):
        time_ns[i] = ts.time / 1000.0
        mob = m["core"].positions
        mc = mob.mean(0)
        R, rms = rotation_matrix(mob - mc, ref_core)
        cols["bb_rmsd"][i] = rms

        def fitted(ag):
            return (ag.positions - mc) @ np.asarray(R).T + r["core"].positions.mean(0)

        h12_xyz, pep_xyz = fitted(m["h12"]), fitted(m["pep_ca"])
        cols["h12_rmsd"][i] = np.sqrt(((h12_xyz - r["h12"].positions) ** 2).sum(1).mean())
        cols["src2_rmsd"][i] = np.sqrt(((pep_xyz - r["pep_ca"].positions) ** 2).sum(1).mean())
        cols["clamp_k362"][i] = distance_array(m["k362"].positions, m["pep_o"].positions).min()
        cols["clamp_e542"][i] = distance_array(m["e542"].positions, m["pep_n"].positions).min()
        if i >= i_eq:
            ca = fitted(m["ca"])
            ca_sum += ca
            ca_sq += (ca ** 2).sum(1)
            n_eq += 1

        if has_lig:
            lx = lig_heavy.positions
            lig_xyz[i] = lx
            cols["lig_rmsd_plain"][i] = np.sqrt(((lx - lig_ref) ** 2).sum(1).mean())
            cog = lx.mean(0)
            cols["lig_pocket_dist"][i] = np.linalg.norm(cog - m["pocket_heavy"].positions.mean(0))
            cols["lig_disp"][i] = np.linalg.norm(cog - lig_ref_cog)
            cols["lig_min_dist"][i] = distance_array(lx, prot_heavy.positions).min()

            if i >= i_eq and (i - i_eq) % L.CONTACT_STRIDE == 0:
                n_sampled += 1
                pairs = capped_distance(lx, prot_heavy.positions, CONTACT_CUT, return_distances=False)
                for rx in {prot_heavy[j].resindex for j in pairs[:, 1]}:
                    cnt["contact"][rx] += 1
                if lig_polar.n_atoms:
                    pp = capped_distance(lig_polar.positions, prot_polar.positions, HB_DIST, return_distances=False)
                    for rx in {prot_polar[j].resindex for j in pp[:, 1]}:
                        cnt["polar"][rx] += 1
                for rx in hbond_residues(u, d_pairs, acc, side_of, u.atoms.positions):
                    cnt["hbond"][rx] += 1
                if lig_np.n_atoms:
                    sp = capped_distance(lig_np.positions, acidic.positions, SALT_CUT, return_distances=False)
                    for rx in {acidic[j].resindex for j in sp[:, 1]}:
                        cnt["saltbridge"][rx] += 1

    ts_df = pd.DataFrame({"time_ns": time_ns, **cols})
    if has_lig:
        Z, adj = heavy_graph(lig_heavy)
        ts_df.insert(1, "lig_rmsd", symm_rmsd(lig_ref, lig_xyz, Z, adj))
        bad = (ts_df.lig_rmsd - ts_df.lig_rmsd_plain > 1e-3).sum()
        assert bad == 0, f"symmetry-corrected RMSD above plain RMSD in {bad} frames"
    else:
        ts_df.insert(1, "lig_rmsd", np.nan)

    mean = ca_sum / n_eq
    rmsf = np.sqrt(np.maximum(ca_sq / n_eq - (mean ** 2).sum(1), 0))
    rmsf_df = pd.DataFrame({"resid": m["ca"].resids, "resname": m["ca"].resnames,
                            "label": [L.label_resid(a.resid, a.resname) for a in m["ca"]], "rmsf": rmsf})

    res = u.residues
    occ = pd.DataFrame({"resid": res.resids, "resname": res.resnames,
                        "label": [L.label_resid(x.resid, x.resname) for x in res],
                        **{k: v / max(n_sampled, 1) for k, v in cnt.items()}})
    occ = occ[(occ.resname != L.LIGAND_RESNAME)]
    occ = occ[(occ[["contact", "polar", "hbond", "saltbridge"]].sum(1) > 0)
              | occ.label.isin(list(L.ANCHORS)) | occ.label.str.match(r"^(Asp351)$")]

    out = L.proc_dir(s, rep)
    ts_df.to_parquet(out / "ts.parquet", index=False)
    rmsf_df.to_csv(out / "rmsf.csv", index=False)
    occ.to_csv(out / "occupancy.csv", index=False)
    if has_lig:                                             # one frame per 0.5 ns from 20 ns: clustering input
        pick = np.arange(i_eq, n - 1, 50)
        np.savez_compressed(out / "lig_frames.npz", xyz=lig_xyz[pick], frame=pick, time_ns=time_ns[pick],
                            ref=lig_ref, Z=Z, adj=adj)
    qc(out / "md.edr", out / "qc.csv", work)
    (out / "metrics.done").write_text(json.dumps({"n_frames": int(n), "n_contact_frames": int(n_sampled)}) + "\n")


def read_xvg(path):
    names, rows = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("@ s") and "legend" in line:
            names.append(line.split("legend", 1)[1].strip().strip('"'))
        elif line and line[0] not in "#@":
            rows.append([float(x) for x in line.split()])
    return pd.DataFrame(rows, columns=["time_ps", *names])


def qc(edr, out_csv, work):
    """Thermodynamic state over 20–100 ns: mean, SD and linear drift of T, P, density, potential, volume."""
    xvg = work / "qc.xvg"
    terms = ["Temperature", "Pressure", "Density", "Potential", "Volume"]
    r = subprocess.run(["gmx", "energy", "-f", str(edr), "-b", str(int(L.T_EQ_NS * 1000)), "-o", str(xvg)],
                       input="\n".join(terms) + "\n\n", capture_output=True, text=True,
                       env=dict(__import__("os").environ, GMX_MAXBACKUP="-1"))
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    df = read_xvg(xvg)
    rows = []
    for t in terms:
        y = df[t].to_numpy()
        slope = np.polyfit(df.time_ps / 1000.0, y, 1)[0]
        rows.append({"term": t, "mean": y.mean(), "sd": y.std(ddof=1),
                     "drift_per_80ns": slope * (L.PROD_NS - L.T_EQ_NS), "n": len(y)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run(s, rep, force=False):
    if (L.proc_dir(s, rep) / "metrics.done").exists() and not force:
        return
    compute(s, rep)
