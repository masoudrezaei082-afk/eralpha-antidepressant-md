"""Stage 5: per-replica values -> per-system statistics, tables (CSV + XLSX) and summary.json.

Every per-system number is a statistic over the three replica means (see stats.py). Frame-level data are only used
inside a replica (time averages from 20 ns on, convergence checks, clustering).
"""
import json
import shutil

import numpy as np
import pandas as pd

from . import layout as L
from . import mmgbsa as MM
from . import stats as S

TABLES = L.OUT / "tables"
RESULTS = L.REPO / "analysis" / "results"
RETAIN_A = 2.0            # pose retained: symmetry-corrected ligand RMSD below 2 Å
CLAMP_A = 4.0             # charge clamp formed: shortest distance below 4 Å
DISSOC_A = 4.0            # ligand out of contact: no ligand–protein heavy-atom pair within 4 Å
CLUSTER_A = 2.0
OLD_FILES = {"estradiol": "Estradiol", "pape1": "PaPE-1", "imipramine": "Imipramine", "s_ketamine": "Ketamine",
             "hnk": "HNK", "r_fluoxetine": "Fluoxetine"}   # old single runs; the old fluoxetine was one enantiomer


def ordered(df, col="sys"):
    order = {s.key: i for i, s in enumerate(L.SYSTEMS)}
    return df.sort_values(col, key=lambda c: c.map(order)).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ loading
def load_ts(keys=None):
    out = []
    for s, r in L.replicas(keys):
        p = L.proc_dir(s, r) / "ts.parquet"
        if p.exists():
            out.append(pd.read_parquet(p).assign(sys=s.key, rep=r))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_csv(name, keys=None):
    out = []
    for s, r in L.replicas(keys):
        p = L.proc_dir(s, r) / name
        if p.exists():
            out.append(pd.read_csv(p).assign(sys=s.key, rep=r))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ structure
def replica_table(ts):
    eq = ts[ts.time_ns >= L.T_EQ_NS]
    g = eq.groupby(["sys", "rep"])
    t = pd.DataFrame({
        "lig_rmsd": g.lig_rmsd.mean(),
        "retention": g.lig_rmsd.apply(lambda x: (x < RETAIN_A).mean() if x.notna().any() else np.nan),
        "lig_pocket_dist": g.lig_pocket_dist.mean(),
        "lig_disp": g.lig_disp.mean(),
        "out_of_contact": g.lig_min_dist.apply(lambda x: (x > DISSOC_A).mean() if x.notna().any() else np.nan),
        "bb_rmsd": g.bb_rmsd.mean(),
        "h12_rmsd": g.h12_rmsd.mean(),
        "src2_rmsd": g.src2_rmsd.mean(),
        "clamp_k362": g.clamp_k362.mean(),
        "clamp_e542": g.clamp_e542.mean(),
        "clamp_formed": g.apply(lambda d: ((d.clamp_k362 < CLAMP_A) & (d.clamp_e542 < CLAMP_A)).mean(),
                                include_groups=False),
    }).reset_index()
    first = ts[ts.time_ns == 0].set_index(["sys", "rep"])["lig_rmsd"].rename("lig_rmsd_t0")
    return ordered(t.merge(first.reset_index(), on=["sys", "rep"], how="left"))


def system_table(rep_tab, cols):
    rows = []
    for key, d in rep_tab.groupby("sys", sort=False):
        row = {"sys": key, "label": L.BY_KEY[key].label}
        for c in cols:
            s = S.summarize(d[c])
            row.update({f"{c}_mean": s["mean"], f"{c}_sem": s["sem"], f"{c}_lo": s["lo"], f"{c}_hi": s["hi"]})
        row["n"] = len(d)
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def occupancy_tables(occ):
    """Per-system mean ± SEM occupancy per residue label (replicas missing a residue count as 0)."""
    if occ.empty:
        return occ
    kinds = [k for k in ("contact", "polar", "hbond", "saltbridge") if k in occ]
    full = occ.pivot_table(index=["sys", "label"], columns="rep", values=kinds,
                           fill_value=0.0)
    rows = []
    for (key, lab), r in full.iterrows():
        row = {"sys": key, "label": lab}
        for kind in kinds:
            v = [r.get((kind, rep), 0.0) for rep in L.REPLICAS]
            s = S.summarize(v)
            row.update({f"{kind}_mean": s["mean"], f"{kind}_sem": s["sem"]})
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def rmsf_table(rmsf):
    g = rmsf.groupby(["sys", "resid", "label"])["rmsf"]
    t = pd.DataFrame({"rmsf_mean": g.mean(), "rmsf_sem": g.std(ddof=1) / np.sqrt(g.count())}).reset_index()
    return ordered(t)


def qc_table(qc):
    return ordered(qc)


# ------------------------------------------------------------------------------------------------ energies
def energy_tables(mm):
    """System means at each method/ε and the difference from estradiol (Welch 95 % CI)."""
    if mm.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, diffs = [], []
    for (key, meth, eps), d in mm.groupby(["sys", "method", "eps"], sort=False):
        s = S.summarize(d.d_TOTAL)
        rows.append({"sys": key, "label": L.BY_KEY[key].label, "method": meth, "eps": eps, **s,
                     "frame_sem_mean": d.frame_sem.mean()})
        ref = mm[(mm.sys == L.REFERENCE) & (mm.method == meth) & (mm.eps == eps)]
        if key != L.REFERENCE and len(ref) >= 2 and len(d) >= 2:
            w = S.welch(d.d_TOTAL, ref.d_TOTAL)
            diffs.append({"sys": key, "label": L.BY_KEY[key].label, "method": meth, "eps": eps, **w})
    sysdf = ordered(pd.DataFrame(rows))
    dd = ordered(pd.DataFrame(diffs)) if diffs else pd.DataFrame()
    if not dd.empty:
        verdict = {k: S.ordering_supported(d.to_dict("records")) for k, d in dd.groupby("sys")}
        dd["ordering_vs_estradiol"] = dd.sys.map(verdict)
    return sysdf, dd


def rank_agreement(sysdf):
    """Spearman ρ between system rankings: GB ε pairs and GB ε2 vs PB ε2 (all liganded systems with data)."""
    out = {}
    piv = sysdf.pivot_table(index="sys", columns=["method", "eps"], values="mean")
    pairs = [(("gb", 1), ("gb", 2)), (("gb", 2), ("gb", 4)), (("gb", 1), ("gb", 4)), (("gb", 2), ("pb", 2))]
    for a, b in pairs:
        if a in piv and b in piv:
            d = piv[[a, b]].dropna()
            if len(d) >= 3:
                rho, p = S.spearman(d[a], d[b])
                out[f"{a[0]}{a[1]}_vs_{b[0]}{b[1]}"] = {"rho": round(rho, 3), "p": round(p, 4), "n_systems": len(d)}
    return out


def old_vs_new(sysdf):
    rows = []
    for key, name in OLD_FILES.items():
        p = L.REPO / "reference" / "old_results" / f"{name}.txt"
        if not p.exists():
            continue
        o = MM.parse_old(p)
        new = sysdf[(sysdf.sys == key) & (sysdf.method == "gb") & (sysdf.eps == 2)]
        newpb = sysdf[(sysdf.sys == key) & (sysdf.method == "pb") & (sysdf.eps == 2)]
        rows.append({"sys": key, "label": L.BY_KEY[key].label, "old_gb": o["gb"], "old_pb": o["pb"],
                     "old_frames": o["n_frames"],
                     "new_gb_eps2_mean": new["mean"].iloc[0] if len(new) else np.nan,
                     "new_gb_eps2_sem": new["sem"].iloc[0] if len(new) else np.nan,
                     "new_pb_eps2_mean": newpb["mean"].iloc[0] if len(newpb) else np.nan,
                     "new_pb_eps2_sem": newpb["sem"].iloc[0] if len(newpb) else np.nan})
    return ordered(pd.DataFrame(rows)) if rows else pd.DataFrame()


def convergence(frames, ts):
    """Within-replica block SEM (Flyvbjerg–Petersen) for ΔE_GB(ε2) and ligand RMSD, plus ΔE first/second half."""
    rows = []
    if not frames.empty:
        for (key, rep), d in frames[(frames.method == "gb") & (frames.eps == 2)].groupby(["sys", "rep"]):
            x = d.sort_values("frame").TOTAL.to_numpy()
            _, plat, conv = S.block_sem(x)
            h = len(x) // 2
            rows.append({"sys": key, "rep": rep, "quantity": "dE_gb_eps2", "n": len(x), "mean": x.mean(),
                         "naive_sem": x.std(ddof=1) / np.sqrt(len(x)), "block_sem": plat, "plateau": conv,
                         "first_half": x[:h].mean(), "second_half": x[h:].mean()})
    for (key, rep), d in ts[ts.time_ns >= L.T_EQ_NS].groupby(["sys", "rep"]):
        x = d.lig_rmsd.to_numpy()
        if np.isnan(x).all():
            continue
        _, plat, conv = S.block_sem(x)
        h = len(x) // 2
        rows.append({"sys": key, "rep": rep, "quantity": "lig_rmsd", "n": len(x), "mean": x.mean(),
                     "naive_sem": x.std(ddof=1) / np.sqrt(len(x)), "block_sem": plat, "plateau": conv,
                     "first_half": x[:h].mean(), "second_half": x[h:].mean()})
    return ordered(pd.DataFrame(rows)) if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ clustering
def cluster_system(s):
    """GROMOS clustering (2 Å, symmetry-corrected, pocket frame) of ligand poses pooled over the replicas, one frame
    per 0.5 ns from 20 ns on; writes the complex of each of the top 3 cluster centroids as PDB."""
    from spyrmsd import rmsd as srmsd
    import MDAnalysis as mda
    from . import traj as TR

    xyz, src = [], []
    for r in L.REPLICAS:
        p = L.proc_dir(s, r) / "lig_frames.npz"
        if not p.exists():
            return None
        z = np.load(p)
        Z, adj = z["Z"], z["adj"]
        xyz += list(z["xyz"])
        src += [(r, int(f), float(t)) for f, t in zip(z["frame"], z["time_ns"])]
    n = len(xyz)
    D = np.zeros((n, n))
    for i in range(n - 1):
        D[i, i + 1:] = srmsd.symmrmsd(xyz[i], xyz[i + 1:], Z, Z, adj, adj, center=False, minimize=False, cache=True)
    D = D + D.T
    left = np.ones(n, bool)
    clusters = []
    while left.any():
        nb = (D < CLUSTER_A) & left[None, :] & left[:, None]
        cnt = nb.sum(1)
        c = int(np.argmax(cnt))
        members = np.where(nb[c])[0]
        clusters.append((c, members))
        left[members] = False
    out = L.ensure(L.OUT / "clusters")
    rows = []
    for k, (c, members) in enumerate(clusters):
        reps = pd.Series([src[m][0] for m in members]).value_counts().to_dict()
        row = {"sys": s.key, "cluster": k + 1, "size": len(members), "fraction": len(members) / n,
               "centroid_rep": src[c][0], "centroid_frame": src[c][1], "centroid_time_ns": src[c][2],
               "replicas": json.dumps({int(a): int(b) for a, b in reps.items()}),
               "n_replicas": len(reps)}
        rows.append(row)
        if k < 3:
            rep, frame = src[c][0], src[c][1]
            d = L.proc_dir(s, rep)
            work = L.ensure(L.WORK / s.rep_id(rep))
            for f in ("complex_ref.gro", "topol_complex.top"):
                if not (work / f).exists():
                    shutil.copyfile(d / f, work / f)
            u = TR.universe(work, d / "processed.xtc")
            u.trajectory[frame]
            u.atoms.write(str(out / f"{s.key}_cluster{k + 1}.pdb"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ driver
def write_tables(tabs, summary):
    for d in (L.ensure(TABLES), L.ensure(RESULTS / "tables")):
        for name, df in tabs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(d / f"{name}.csv", index=False, float_format="%.4g")
        with pd.ExcelWriter(d / "md_tables.xlsx") as xw:
            for name, df in tabs.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(xw, sheet_name=name[:31], index=False)
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


def main_table(struct_sys, esys, dd, occ_sys):
    """One row per system: ΔE_GB ε1/2/4 and ΔE_PB ε2 (mean ± SEM), ΔΔE_GB(ε2) vs estradiol [95 % CI], pose
    retention and anchor H-bond occupancy."""
    rows = []
    for s in L.SYSTEMS:
        row = {"System": s.label}
        for meth, eps in (("gb", 1), ("gb", 2), ("gb", 4), ("pb", 2)):
            e = esys[(esys.sys == s.key) & (esys.method == meth) & (esys.eps == eps)] if not esys.empty else []
            row[f"ΔE_{meth.upper()} ε{eps}"] = (f"{e['mean'].iloc[0]:.1f} ± {e['sem'].iloc[0]:.1f}"
                                               if len(e) else "")
        d = dd[(dd.sys == s.key) & (dd.method == "gb") & (dd.eps == 2)] if not dd.empty else []
        row["ΔΔE_GB ε2 vs E2 [95% CI]"] = (f"{d['diff'].iloc[0]:+.1f} [{d['lo'].iloc[0]:+.1f}, {d['hi'].iloc[0]:+.1f}]"
                                          if len(d) else ("ref." if s.key == L.REFERENCE else ""))
        st_ = struct_sys[struct_sys.sys == s.key]
        if s.ligand and len(st_) and not np.isnan(st_["retention_mean"].iloc[0]):
            row["Pose retention (%)"] = f"{100 * st_['retention_mean'].iloc[0]:.0f} ± {100 * st_['retention_sem'].iloc[0]:.0f}"
        for anchor in L.ANCHORS:
            o = occ_sys[(occ_sys.sys == s.key) & (occ_sys.label == anchor)] if not occ_sys.empty else []
            row[f"H-bond {anchor} (%)"] = f"{100 * o['hbond_mean'].iloc[0]:.0f}" if len(o) else ("" if not s.ligand
                                                                                                  else "0")
        rows.append(row)
    return pd.DataFrame(rows)


def structure_vs_estradiol(rep_tab):
    """Ligand RMSD and pose retention minus estradiol's, per system: Welch difference of replica means with 95 % CI."""
    rows = []
    ref = rep_tab[rep_tab.sys == L.REFERENCE]
    for s in L.LIGANDED:
        if s.key == L.REFERENCE:
            continue
        d = rep_tab[rep_tab.sys == s.key]
        for col in ("lig_rmsd", "retention"):
            if d[col].notna().sum() >= 2:
                rows.append({"sys": s.key, "label": s.label, "quantity": col, **S.welch(d[col], ref[col])})
    return pd.DataFrame(rows)


def run(cluster=True):
    ts = load_ts()
    rep_tab = replica_table(ts)
    struct_cols = ["lig_rmsd", "retention", "lig_pocket_dist", "out_of_contact", "bb_rmsd", "h12_rmsd", "src2_rmsd",
                   "clamp_k362", "clamp_e542", "clamp_formed", "lig_rmsd_t0"]
    struct_sys = system_table(rep_tab, struct_cols)
    occ = load_csv("occupancy.csv")
    occ_sys = occupancy_tables(occ)
    rmsf = rmsf_table(load_csv("rmsf.csv"))
    qc = qc_table(load_csv("qc.csv"))
    mm, frames, decomp = MM.collect()
    esys, dd = energy_tables(mm)
    ranks = rank_agreement(esys) if not esys.empty else {}
    old = old_vs_new(esys) if not esys.empty else pd.DataFrame()
    conv = convergence(frames, ts)
    dec_sys = pd.DataFrame()
    if not decomp.empty:
        g = decomp[decomp.side == "R"].groupby(["sys", "resname", "resnum"])["TOTAL"]
        dec_sys = ordered(pd.DataFrame({"mean": g.mean(), "sem": g.std(ddof=1) / np.sqrt(g.count()),
                                        "n": g.count()}).reset_index())
    clus = pd.DataFrame()
    if cluster:
        parts = [cluster_system(s) for s in L.LIGANDED]
        parts = [p for p in parts if p is not None]
        clus = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    tabs = {"main_table": main_table(struct_sys, esys, dd, occ_sys),
            "structure_per_replica": rep_tab, "structure_per_system": struct_sys,
            "energy_per_replica": ordered(mm) if not mm.empty else mm, "energy_per_system": esys,
            "ddE_vs_estradiol": dd, "old_vs_new": old, "occupancy": occ_sys, "rmsf": rmsf, "qc": qc,
            "convergence": conv, "decomp_eps2": dec_sys, "clusters": clus,
            "structure_vs_estradiol": structure_vs_estradiol(rep_tab)}
    summary = {
        "n_replicas_with_metrics": int(rep_tab[["sys", "rep"]].drop_duplicates().shape[0]),
        "n_energy_jobs": int(len(mm)),
        "rank_agreement": ranks,
        "ordering_vs_estradiol": (dd.groupby("sys").ordering_vs_estradiol.first().to_dict() if not dd.empty else {}),
        "thresholds": {"retention_A": RETAIN_A, "clamp_A": CLAMP_A, "out_of_contact_A": DISSOC_A,
                       "cluster_A": CLUSTER_A, "equilibration_ns": L.T_EQ_NS},
    }
    write_tables(tabs, summary)
    return tabs, summary
