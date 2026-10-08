"""Build every simulation system listed in prep/systems.yaml, non-interactively.

    python prep/build_system.py [system ...]        # default: all systems

Prerequisite: python prep/build_receptor.py (writes receptor/receptor_wt_src2.pdb).

Per system  systems/<name>/
    params/lig.mol2, lig.frcmod      GAFF2 types + AM1-BCC charges, computed once from the first pose,
                                     so every pose and replica of a ligand has identical parameters
    <pose>/ligand.sdf, ligand.mol2   the placed ligand (all H, template atom order)
    <pose>/build/                    tleap input/log, Amber prmtop/inpcrd, GROMACS topol.top, conf.gro,
                                     posre_*.itp, index.ndx (gitignored; rebuilt by this script, shipped to Drive)
    provenance.yaml                  SMILES, InChIKey, CIP labels, charge, pose source and fit, box, ions

System: ff14SB protein + GAFF2 ligand + TIP3P water + Joung-Cheatham ions, truncated octahedron with a
12 Å solute-wall buffer, neutralised, then 0.15 M NaCl (ion pairs = 0.15 M x N_water / 55.5 M).
"""
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import parmed as pmd
import yaml
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

sys.path.insert(0, str(Path(__file__).parent))
import ligands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "prep" / "systems.yaml"))
RECEPTOR = ROOT / "receptor" / "receptor_wt_src2.pdb"
BUFFER_A = 12.0
SALT_M = 0.15
SOLVENT_RES = {"WAT", "HOH", "Na+", "Cl-", "NA", "CL"}


def run(cmd, cwd, log=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if log:
        Path(cwd, log).write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed in {cwd}:\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout + r.stderr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- ligand
def template(sd):
    if sd["smiles"] == "from_pose":
        return ligands.template_from_allH_file(ROOT / sd["smiles_source"], sd["charge"])
    return ligands.template_from_smiles(sd["smiles"])


def place(sd, tmpl, pose):
    if pose["kind"] == "glide":
        heavy, info = ligands.best_glide(tmpl, ROOT, pose["cid"])
    else:
        heavy, info = ligands.load_pose(pose, ROOT, sd)
    mol, fit = ligands.place_on_pose(tmpl, heavy)
    return mol, {**info, **fit, "final_cip": ligands.cip_text(mol)}


def parametrise(mol, charge, pdir):
    pdir.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(pdir / "lig_input.sdf"))
    run(["antechamber", "-i", "lig_input.sdf", "-fi", "sdf", "-o", "lig.mol2", "-fo", "mol2",
         "-c", "bcc", "-nc", str(charge), "-at", "gaff2", "-rn", "LIG", "-pf", "y", "-dr", "no"],
        pdir, "antechamber.log")
    run(["parmchk2", "-i", "lig.mol2", "-f", "mol2", "-o", "lig.frcmod", "-s", "gaff2"], pdir)
    frc = (pdir / "lig.frcmod").read_text()
    residual = make_integral(pdir / "lig.mol2", charge)
    q = sum(float(l.split()[8]) for l in mol2_atoms(pdir / "lig.mol2"))
    return {"atom_types": "GAFF2", "charges": "AM1-BCC (antechamber/sqm)",
            "charge_geometry": "first pose listed in systems.yaml",
            "sum_of_charges": round(q, 6),
            "rounding_residual_spread_e": residual,
            "frcmod_terms_needing_revision": frc.count("ATTN, need revision")}


def make_integral(mol2, charge):
    """antechamber rounds AM1-BCC charges, leaving the sum ~1e-3 e off the formal charge. Spread the
    residual evenly (in exact 1e-6 e units) so the ligand, and hence the system, is integral."""
    lines = open(mol2).read().splitlines(keepends=True)
    rows, on = [], False
    for k, l in enumerate(lines):
        if l.startswith("@<TRIPOS>"):
            on = l.strip() == "@<TRIPOS>ATOM"
        elif on and l.strip():
            rows.append(k)
    q = [round(float(lines[k].split()[8]) * 1e6) for k in rows]
    excess = sum(q) - charge * 10 ** 6
    n = len(q)
    q = [x - excess // n for x in q]
    for k in range(excess % n):
        q[k] -= 1
    assert sum(q) == charge * 10 ** 6
    for k, x in zip(rows, q):
        f = lines[k].split()
        lines[k] = (f"{int(f[0]):7d} {f[1]:<8s}{float(f[2]):10.4f}{float(f[3]):10.4f}{float(f[4]):10.4f} "
                    f"{f[5]:<6s}{int(f[6]):4d} {f[7]:<6s}{x / 1e6:10.6f}\n")
    Path(mol2).write_text("".join(lines))
    return round(excess / 1e6, 6)


def mol2_atoms(path):
    lines, on = [], False
    for l in open(path):
        if l.startswith("@<TRIPOS>"):
            on = l.strip() == "@<TRIPOS>ATOM"
            continue
        if on and l.strip():
            lines.append(l)
    return lines


def mol2_with_coords(template_mol2, mol, out):
    """The parametrised mol2 with this pose's coordinates (atom order is identical by construction)."""
    pos = mol.GetConformer().GetPositions()
    src = open(template_mol2).read().splitlines(keepends=True)
    i, on, res = 0, False, []
    for l in src:
        if l.startswith("@<TRIPOS>"):
            on = l.strip() == "@<TRIPOS>ATOM"
        elif on and l.strip():
            f = l.split()
            el = re.sub(r"[^A-Za-z]", "", f[1])[:1]
            assert mol.GetAtomWithIdx(i).GetSymbol()[0].upper() == el.upper(), "atom order mismatch"
            x, y, z = pos[i]
            l = f"{int(f[0]):7d} {f[1]:<8s}{x:10.4f}{y:10.4f}{z:10.4f} {f[5]:<6s}{int(f[6]):4d} {f[7]:<6s}{float(f[8]):10.6f}\n"
            i += 1
        res.append(l)
    assert i == mol.GetNumAtoms()
    Path(out).write_text("".join(res))


# ---------------------------------------------------------------- tleap / parmed
LEAP_HEAD = """source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.gaff2
"""


def leap_script(has_lig, ions=None):
    s = LEAP_HEAD
    if has_lig:
        s += "loadamberparams ../../params/lig.frcmod\nLIG = loadmol2 ../ligand.mol2\n"
    s += f"rec = loadpdb {RECEPTOR}\n"
    s += "complex = combine {rec LIG}\n" if has_lig else "complex = rec\n"
    s += f"solvateOct complex TIP3PBOX {BUFFER_A}\n"
    if ions is None:
        s += "savepdb complex solvated.pdb\n"
    else:
        s += "addIonsRand complex Na+ 0\naddIonsRand complex Cl- 0\n"
        if ions:
            s += f"addIonsRand complex Na+ {ions} Cl- {ions}\n"
        s += "charge complex\nsaveamberparm complex complex.prmtop complex.inpcrd\nsavepdb complex complex.pdb\n"
    return s + "quit\n"


def tleap(bdir, has_lig):
    (bdir / "count.in").write_text(leap_script(has_lig))
    run(["tleap", "-f", "count.in"], bdir, "count.log")
    n_wat = sum(1 for l in open(bdir / "solvated.pdb") if l[12:16].strip() == "O" and l[17:20] == "WAT")
    pairs = round(SALT_M * n_wat / 55.5)
    (bdir / "tleap.in").write_text(leap_script(has_lig, pairs))
    out = run(["tleap", "-f", "tleap.in"], bdir, "tleap.log")
    (bdir / "solvated.pdb").unlink()
    warn = [l for l in out.splitlines() if "WARNING" in l.upper() and "residue" not in l.lower()]
    return n_wat, pairs, warn


def to_gromacs(bdir):
    s = pmd.load_file(str(bdir / "complex.prmtop"), str(bdir / "complex.inpcrd"))
    s.save(str(bdir / "topol.top"), format="gromacs", overwrite=True)
    s.save(str(bdir / "conf.gro"), overwrite=True)
    add_posres(bdir / "topol.top")
    groups = write_index(s, bdir / "index.ndx")
    counts = {r: sum(1 for x in s.residues if x.name == r) for r in ("WAT", "Na+", "Cl-")}
    a, al = float(s.box[0]), float(s.box[3])
    return {"atoms": len(s.atoms), "waters": counts["WAT"], "Na+": counts["Na+"], "Cl-": counts["Cl-"],
            "net_charge": round(float(sum(x.charge for x in s.atoms)), 6),
            "box": {"shape": "truncated octahedron", "a_A": round(a, 2), "angles_deg": round(al, 2),
                    "volume_nm3": round(4 / (3 * 3 ** 0.5) * (a / 10) ** 3, 1)},
            "index_groups": groups}


def add_posres(top):
    """Heavy-atom position restraints for every non-solvent molecule type, switched by -DPOSRES and
    scaled by POSRES_FC (kJ/mol/nm2, set in each mdp's `define`)."""
    text = top.read_text().splitlines(keepends=True)
    out, mol, atoms, section = [], None, [], None

    def flush():
        if mol and atoms and not {r for _, r, _ in atoms} <= SOLVENT_RES:
            heavy = [n for n, _, m in atoms if m > 1.5]
            itp = top.parent / f"posre_{mol}.itp"
            itp.write_text("#ifndef POSRES_FC\n#define POSRES_FC 1000\n#endif\n[ position_restraints ]\n"
                           ";  ai  funct  fcx        fcy        fcz\n" +
                           "".join(f"{n:6d}     1  POSRES_FC  POSRES_FC  POSRES_FC\n" for n in heavy))
            out.append(f'\n#ifdef POSRES\n#include "posre_{mol}.itp"\n#endif\n\n')

    for l in text:
        s = l.strip()
        if s.startswith("["):
            section = s.strip("[] ").lower()
            if section in ("moleculetype", "system"):
                flush()
                mol, atoms = None, []
        elif section == "moleculetype" and s and not s.startswith(";") and mol is None:
            mol = s.split()[0]
        elif section == "atoms" and s and not s.startswith((";", "#")):
            f = s.split()
            atoms.append((int(f[0]), f[3], float(f[7])))
        out.append(l)
    top.write_text("".join(out))


def write_index(s, path):
    lig = [a.idx + 1 for a in s.atoms if a.residue.name == "LIG"]
    rec = [a.idx + 1 for a in s.atoms if a.residue.name not in SOLVENT_RES and a.residue.name != "LIG"]
    groups = {"System": list(range(1, len(s.atoms) + 1)), "Receptor": rec, "Ligand": lig,
              "Complex": sorted(rec + lig),
              "Solvent_Ions": [a.idx + 1 for a in s.atoms if a.residue.name in SOLVENT_RES]}
    with open(path, "w") as fh:
        for name, idx in groups.items():
            if not idx:
                continue
            fh.write(f"[ {name} ]\n")
            for k in range(0, len(idx), 15):
                fh.write(" ".join(f"{i:6d}" for i in idx[k:k + 15]) + "\n")
    return {k: len(v) for k, v in groups.items() if v}


def grompp_check(bdir):
    """grompp EM and NVT inputs with -maxwarn 0: any warning is a build failure."""
    res = {}
    for stage in ("em", "nvt"):
        mdp = bdir / f"check_{stage}.mdp"
        mdp.write_text((ROOT / "mdp" / f"{stage}.mdp").read_text().replace("@SEED@", "1"))
        out = run(["gmx", "grompp", "-f", mdp.name, "-c", "conf.gro", "-r", "conf.gro", "-p", "topol.top",
                   "-n", "index.ndx", "-o", f"check_{stage}.tpr", "-po", f"check_{stage}_out.mdp",
                   "-maxwarn", "0"], bdir, f"grompp_{stage}.log")
        notes = len(re.findall(r"^NOTE \d+", out, flags=re.M))
        res[stage] = {"warnings": 0, "notes": notes}
        for f in bdir.glob(f"check_{stage}*"):
            f.unlink()
    return res


# ---------------------------------------------------------------- driver
def build(name, sd):
    sdir = ROOT / "systems" / name
    sdir.mkdir(parents=True, exist_ok=True)
    prov = {"system": name, "receptor": {"file": "receptor/receptor_wt_src2.pdb", "sha256_16": sha(RECEPTOR),
                                         "details": "receptor/receptor_prep.yaml"},
            "force_field": {"protein": "ff14SB", "ligand": "GAFF2 + AM1-BCC", "water": "TIP3P",
                            "ions": "Joung-Cheatham (TIP3P)"},
            "solvation": {"box": "truncated octahedron", "buffer_A": BUFFER_A,
                          "salt": f"neutralised + {SALT_M} M NaCl"}}
    has_lig = bool(sd.get("smiles"))
    tmpl = None
    if has_lig:
        tmpl = template(sd)
        prov["ligand"] = {"name": sd["name"], "smiles_simulated": Chem.MolToSmiles(tmpl),
                          "inchikey": MolToInchiKey(tmpl), "formal_charge": Chem.GetFormalCharge(tmpl),
                          "cip": ligands.cip_text(tmpl), "pubchem_cid": sd.get("pubchem_cid"),
                          "note": sd.get("note")}
        if sd.get("source"):
            prov["ligand"]["source"] = sd["source"]
        want = sorted(ligands.cip(tmpl).values())
        if "cip" in sd and sorted(sd["cip"]) != want:
            raise ValueError(f"{name}: template CIP {want} != systems.yaml {sd['cip']}")
        assert Chem.GetFormalCharge(tmpl) == sd["charge"], f"{name}: charge mismatch"
    prov["poses"] = {}
    for k, pose in enumerate(sd["poses"]):
        pdir = sdir / pose["id"]
        bdir = pdir / "build"
        bdir.mkdir(parents=True, exist_ok=True)
        entry = {"replicas": pose["replicas"]}
        if has_lig:
            mol, fit = place(sd, tmpl, pose)
            Chem.MolToMolFile(mol, str(pdir / "ligand.sdf"))
            if k == 0:
                prov["ligand"]["parameters"] = parametrise(mol, sd["charge"], sdir / "params")
            mol2_with_coords(sdir / "params" / "lig.mol2", mol, pdir / "ligand.mol2")
            entry["pose"] = fit
        n_wat, pairs, warn = tleap(bdir, has_lig)
        entry["salt_pairs_added"] = pairs
        entry["tleap_warnings"] = warn
        entry["system"] = to_gromacs(bdir)
        entry["grompp"] = grompp_check(bdir)
        if abs(entry["system"]["net_charge"]) > 1e-3:
            raise ValueError(f"{name}/{pose['id']}: net charge {entry['system']['net_charge']}")
        prov["poses"][pose["id"]] = entry
        print(f"  {name}/{pose['id']}: {entry['system']['atoms']} atoms, {pairs} NaCl pairs, "
              f"grompp {entry['grompp']}", flush=True)
    yaml.safe_dump(prov, open(sdir / "provenance.yaml", "w"), sort_keys=False, width=110)


def main():
    if not RECEPTOR.exists():
        sys.exit("run prep/build_receptor.py first")
    for tool in ("tleap", "antechamber", "parmchk2", "gmx"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH (use the tools environment)")
    names = sys.argv[1:] or list(CFG["systems"])
    for name in names:
        print(f"== {name}", flush=True)
        build(name, CFG["systems"][name])


if __name__ == "__main__":
    main()
