"""Stage 1-2: join the production parts of a replica and make an analysis-ready trajectory.

For each replica:
  1. copy md.part*.{xtc,edr} from Drive (runs/ only), join them with gmx trjcat / eneconv (a part cut short by a crash
     overlaps the next one; both tools keep the later file's frame), check 10,001 frames at 10 ps from 0 to 100 ns;
  2. build a complex-only topology (receptor + SRC2 + ligand: the xtc holds the Complex group only) and a matching
     tpr/ndx for gmx_MMPBSA;
  3. make every molecule whole, put SRC2 and the ligand in the periodic image nearest the receptor (triclinic minimum
     image: the box is a truncated octahedron), and fit each frame on the pocket Cα atoms of the starting structure.
The processed trajectory is written to Drive analysis/replicas/<id>/processed.xtc.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import layout as L


def sh(cmd, stdin=None, cwd=None):
    env = dict(os.environ, GMX_MAXBACKUP="-1")            # no #file.N# backups of overwritten outputs
    r = subprocess.run(cmd, input=stdin, cwd=cwd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return r.stdout + r.stderr


def read_ndx(path):
    groups, cur = {}, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            cur = line.strip("[] ")
            groups[cur] = []
        elif line and cur:
            groups[cur] += [int(x) for x in line.split()]
    return groups


def write_ndx(path, groups):
    with open(path, "w") as f:
        for name, idx in groups.items():
            f.write(f"[ {name} ]\n")
            for i in range(0, len(idx), 15):
                f.write(" ".join(str(x) for x in idx[i:i + 15]) + "\n")


def trim_top(src, dst, keep=("system1", "system2", "LIG")):
    """Copy a self-contained GROMACS topology, keeping only the complex molecules in [ molecules ]."""
    out, section = [], None
    for line in Path(src).read_text().splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[] ").lower()
        elif section == "molecules" and s and not s.startswith(";"):
            if s.split()[0] not in keep:
                continue
        out.append(line)
    Path(dst).write_text("\n".join(out) + "\n")


def complex_gro(conf, dst, n_atoms):
    """The first n_atoms of conf.gro (the Complex group is atoms 1..n) with the original box line."""
    lines = Path(conf).read_text().splitlines()
    Path(dst).write_text("\n".join([lines[0], f"{n_atoms:5d}", *lines[2:2 + n_atoms], lines[-1]]) + "\n")


def assemble(s, rep, work):
    """Copy and join the parts; returns the joined xtc and edr paths (local)."""
    data = s.run_dir(rep) / "data"
    raw = L.ensure(work / "raw")
    xtcs = sorted(data.glob("md.part*.xtc"))
    edrs = sorted(data.glob("md.part*.edr"))
    if not xtcs or len(xtcs) != len(edrs):
        raise RuntimeError(f"{s.rep_id(rep)}: {len(xtcs)} xtc vs {len(edrs)} edr parts in {data}")
    if not (s.run_dir(rep) / "state" / "DONE").exists():
        raise RuntimeError(f"{s.rep_id(rep)}: no state/DONE on Drive")
    for f in xtcs + edrs:
        dst = raw / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copyfile(f, dst)
    joined_xtc, joined_edr = work / "joined.xtc", work / "joined.edr"
    sh(["gmx", "trjcat", "-f", *[str(raw / f.name) for f in xtcs], "-o", str(joined_xtc)])
    sh(["gmx", "eneconv", "-f", *[str(raw / f.name) for f in edrs], "-o", str(joined_edr)])
    return joined_xtc, joined_edr, [f.name for f in xtcs]


def complex_inputs(s, rep, work):
    """Complex-only topology, reference structure, tpr and index (gmx_MMPBSA groups: 0 Receptor, 1 Ligand)."""
    inp = s.input_dir()
    groups = read_ndx(inp / "index.ndx")
    cpx = groups["Complex"]
    assert cpx == list(range(1, len(cpx) + 1)), "Complex group is expected to be atoms 1..N"
    n = len(cpx)
    trim_top(inp / "topol.top", work / "topol_complex.top")
    complex_gro(inp / "conf.gro", work / "complex_ref.gro", n)
    sh(["gmx", "convert-tpr", "-s", str(s.run_dir(rep) / "state" / "md.tpr"), "-n", str(inp / "index.ndx"),
        "-o", str(work / "complex.tpr")], stdin="Complex\n")
    ndx = {"Receptor": groups["Receptor"]}
    if s.ligand:
        ndx["Ligand"] = groups["Ligand"]
    write_ndx(work / "complex.ndx", ndx)
    return n


def element_of(mass):
    table = {1: "H", 12: "C", 14: "N", 16: "O", 19: "F", 32: "S", 35: "Cl", 80: "Br", 127: "I", 31: "P"}
    return table.get(int(round(mass)), "X")


def universe(work, traj=None):
    """Complex Universe: names/resids from complex_ref.gro, bonds/charges/masses from the trimmed topology."""
    import MDAnalysis as mda
    top = mda.Universe(str(work / "topol_complex.top"), topology_format="ITP")
    u = mda.Universe(str(work / "complex_ref.gro"), *( [str(traj)] if traj else []))
    if top.atoms.n_atoms != u.atoms.n_atoms or list(top.atoms.names) != list(u.atoms.names):
        raise RuntimeError("topology and complex_ref.gro disagree on atoms")
    u.add_TopologyAttr("charges", top.atoms.charges)
    u.atoms.masses = top.atoms.masses
    u.add_TopologyAttr("elements", [element_of(m) for m in top.atoms.masses])
    u.add_bonds(top.bonds.to_indices())
    return u


def selections(u, ref, ligand_ref_sel=None):
    """Atom groups used everywhere. The pocket is every protein residue with a heavy atom within 6 Å of the ligand
    in the starting structure (apo: the estradiol pocket, given by residue ids)."""
    rec = f"resid {L.REC_RESID[0]}:{L.REC_RESID[1]}"
    pep = f"resid {L.PEP_RESID[0]}:{L.PEP_RESID[1]}"
    if ligand_ref_sel is None:
        pocket_resids = sorted({a.resid for a in ref.select_atoms(
            f"({rec} or {pep}) and not name H* and around 6.0 (resname {L.LIGAND_RESNAME} and not name H*)")})
    else:
        pocket_resids = ligand_ref_sel
    return {
        "rec": rec, "pep": pep,
        "pocket_resids": pocket_resids,
        "pocket_ca": "name CA and resid " + " ".join(map(str, pocket_resids)),
        "ligand": f"resname {L.LIGAND_RESNAME}",
    }


def make_processed(work, n_frames_expected=L.N_FRAMES, pocket_resids=None):
    """Whole molecules, nearest image to the receptor, pocket-Cα fit onto the starting structure."""
    import MDAnalysis as mda
    from MDAnalysis import transformations as T
    from MDAnalysis.lib.distances import minimize_vectors

    # whole molecules with gmx (fast, uses the tpr bonds); imaging and fitting below are cheap numpy work
    sh(["gmx", "trjconv", "-s", str(work / "complex.tpr"), "-f", str(work / "joined.xtc"), "-pbc", "mol",
        "-o", str(work / "whole.xtc")], stdin="0\n")
    u = universe(work, work / "whole.xtc")
    ref = universe(work)
    sel = selections(u, ref, pocket_resids)

    times = np.array([ts.time for ts in u.trajectory[[0, -1]]])
    n = u.trajectory.n_frames
    dt = u.trajectory.dt
    checks = {"n_frames": n, "t0_ps": float(times[0]), "t_end_ps": float(times[1]), "dt_ps": float(dt)}
    if n != n_frames_expected or abs(times[0]) > 1e-3 or abs(times[1] - L.PROD_NS * 1000) > 1e-3 \
            or abs(dt - L.DT_PS) > 1e-6:
        raise RuntimeError(f"joined trajectory is not 0-{L.PROD_NS} ns at {L.DT_PS} ps: {checks}")

    frags = u.atoms.fragments
    rec_frag = u.select_atoms(sel["rec"]).fragments[0]
    others = [f for f in frags if f.ix[0] != rec_frag.ix[0]]

    def nearest_image(ts):
        rc = rec_frag.center_of_geometry()
        for f in others:
            d = f.center_of_geometry() - rc
            dmin = minimize_vectors(d[None, :], ts.dimensions)[0]
            f.positions = f.positions + (dmin - d)
        return ts

    mobile_fit = u.select_atoms(sel["pocket_ca"])
    ref_fit = ref.select_atoms(sel["pocket_ca"])
    assert mobile_fit.n_atoms == ref_fit.n_atoms > 10, "pocket Cα selection is too small"
    u.trajectory.add_transformations(nearest_image, T.fit_rot_trans(mobile_fit, ref_fit))

    out = work / "processed.xtc"
    lig = u.select_atoms(f"{sel['ligand']} and not name H*")
    prot_heavy = u.select_atoms(f"({sel['rec']} or {sel['pep']}) and not name H*")
    rec_heavy = u.select_atoms(f"{sel['rec']} and not name H*")
    pep_heavy = u.select_atoms(f"{sel['pep']} and not name H*")
    from MDAnalysis.lib.distances import distance_array
    worst_lig, worst_pep = 0.0, 0.0
    with mda.Writer(str(out), u.atoms.n_atoms) as w:
        for i, ts in enumerate(u.trajectory):
            w.write(u.atoms)
            if i % 50 == 0:                                   # PBC sanity: nothing left in a far image
                worst_pep = max(worst_pep, distance_array(pep_heavy.positions, rec_heavy.positions).min())
                if lig.n_atoms:
                    worst_lig = max(worst_lig, distance_array(lig.positions, prot_heavy.positions).min())
    checks.update({"max_min_dist_ligand_protein_A": round(float(worst_lig), 2),
                   "max_min_dist_src2_receptor_A": round(float(worst_pep), 2),
                   "pocket_resids": [int(r) for r in sel["pocket_resids"]],
                   "n_atoms": int(u.atoms.n_atoms)})
    return out, checks


def process(s, rep, force=False):
    """Stages 1-2 for one replica; results go to Drive analysis/replicas/<id>/."""
    dst = L.ensure(L.proc_dir(s, rep))
    if (dst / "processed.done").exists() and not force:
        return json.loads((dst / "traj_checks.json").read_text())
    work = L.ensure(L.WORK / s.rep_id(rep))
    _, _, parts = assemble(s, rep, work)
    complex_inputs(s, rep, work)
    pocket = None
    if not s.ligand:      # apo: fit on the estradiol pocket so apo and liganded receptors are comparable
        est = L.proc_dir(L.BY_KEY[L.REFERENCE], 1) / "traj_checks.json"
        if not est.exists():
            raise RuntimeError("process estradiol_rep1 before apo (apo uses the estradiol pocket)")
        pocket = json.loads(est.read_text())["pocket_resids"]
    _, checks = make_processed(work, pocket_resids=pocket)
    checks["parts"] = parts
    for f in ("processed.xtc", "complex_ref.gro", "complex.tpr", "complex.ndx", "topol_complex.top"):
        shutil.copyfile(work / f, dst / f)
    shutil.copyfile(work / "joined.edr", dst / "md.edr")
    (dst / "traj_checks.json").write_text(json.dumps(checks, indent=1))
    (dst / "processed.done").write_text("ok\n")
    shutil.rmtree(work / "raw", ignore_errors=True)
    for f in ("joined.xtc", "whole.xtc"):
        (work / f).unlink(missing_ok=True)
    return checks
