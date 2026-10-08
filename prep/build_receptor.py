"""Build the wild-type ERα LBD + SRC2 coactivator receptor from PDB 5WGD.

Output: receptor/receptor_wt_src2.pdb (heavy atoms, AMBER residue names, ready for tleap) and
receptor/receptor_prep.yaml (every decision taken, for docs/methods_text.md).

Steps
 1. 5WGD chain A (residues 308-547; altloc A) + the SRC2 peptide chain in contact with chain A's AF-2
    groove + crystal estradiol (used only to steer His tautomers). Crystal waters are dropped.
 2. The disordered 461-472 loop (absent from the crystal) is grafted from the SWISS-MODEL model
    built on the same template (same frame; Cα RMSD 0.08 Å to 5WGD).
 3. PDBFixer: S537Y reverts the engineered mutation to wild type; missing side-chain atoms are rebuilt;
    ACE/NME caps are added to chain A (residues 294-307 and 548-554 are not modelled, so charged
    termini would be artificial). The peptide keeps its crystal ACE and C-terminal amide (NHE).
 4. pdb2pqr 3.6 / PROPKA 3.5 at pH 7.4 with estradiol as a ligand: His tautomers (H-bond network
    optimisation) and any Asp/Glu/Lys/His whose pKa crosses 7.4.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent))
import ligands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "receptor"
WORK = ROOT / "scratch" / "receptor"
LOOP = (461, 472)                       # crystal numbering
SWISS_OFFSET = 293                      # SWISS-MODEL residue i == crystal residue i + 293
PH = 7.4


def atoms(path):
    out = []
    for l in open(path):
        if l.startswith(("ATOM", "HETATM")) and l[16] in " A":  # altloc A or none
            out.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), chain=l[21],
                            num=int(l[22:26]), x=float(l[30:38]), y=float(l[38:46]), z=float(l[46:54]),
                            el=ligands._element(l)))
    return out


def xyz(a):
    return np.array([a["x"], a["y"], a["z"]])


def write_pdb(path, groups):
    """groups: list of atom lists; a TER is written after each."""
    n = 0
    with open(path, "w") as fh:
        for g in groups:
            for a in g:
                n += 1
                name = a["name"] if len(a["name"]) == 4 else " " + a["name"]
                fh.write(f"{'ATOM':<6}{n:5d} {name:<4} {a['res']:>3} {a['chain']}{a['num']:4d}    "
                         f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00          {a['el']:>2}\n")
            fh.write("TER\n")
        fh.write("END\n")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    log = {"source": "PDB 5WGD (RCSB download, 1.80 Å, ERα LBD Y537S + estradiol + SRC2 peptides)"}
    crys = atoms(REC / "5wgd_rcsb.pdb")
    chain_a = [a for a in crys if a["chain"] == "A" and a["rec"] == "ATOM"]
    est = [a for a in crys if a["chain"] == "A" and a["res"] == "EST"]

    # 1. SRC2 peptide bound to chain A: the peptide chain with most heavy-atom contacts (< 4.5 Å)
    ca_xyz = np.array([xyz(a) for a in chain_a])
    contacts = {}
    for c in "EF":
        pep = np.array([xyz(a) for a in crys if a["chain"] == c and a["res"] != "HOH"])
        d = np.linalg.norm(pep[:, None] - ca_xyz[None], axis=2)
        contacts[c] = int((d.min(axis=1) < 4.5).sum())
    pep_chain = max(contacts, key=contacts.get)
    pep = [a for a in crys if a["chain"] == pep_chain and a["res"] not in ("HOH", "EST")]  # 5WGD files waters under E
    log["src2_peptide"] = {"chain": pep_chain, "atoms_in_contact_with_chain_A": contacts,
                           "sequence": "ACE-HKILHKLLQDS-NH2" if pep_chain == "E" else "ACE-AILHKLLQDS-NH2"}

    # 2. graft the 461-472 loop from SWISS-MODEL
    swiss = [a for a in atoms(REC / "swissmodel_Y537S_input.pdb")
             if LOOP[0] <= a["num"] + SWISS_OFFSET <= LOOP[1] and a["el"] != "H"]
    for a in swiss:
        a.update(num=a["num"] + SWISS_OFFSET, chain="A", rec="ATOM")
    chain_a = sorted(chain_a + swiss, key=lambda a: a["num"])

    def at(num, name):
        return next(xyz(a) for a in chain_a if a["num"] == num and a["name"] == name)
    log["loop_graft"] = {
        "residues": f"{LOOP[0]}-{LOOP[1]}", "source": "receptor/swissmodel_Y537S_input.pdb",
        "junction_C_N_A": [round(float(np.linalg.norm(at(LOOP[0] - 1, "C") - at(LOOP[0], "N"))), 2),
                           round(float(np.linalg.norm(at(LOOP[1], "C") - at(LOOP[1] + 1, "N"))), 2)]}

    # peptide C-terminal NH2 is not a PDBFixer residue: set it aside, re-attach as AMBER NHE
    nh2 = [a for a in pep if a["res"] == "NH2"]
    pep_core = [a for a in pep if a["res"] != "NH2"]
    raw = WORK / "01_merged.pdb"
    write_pdb(raw, [chain_a, pep_core])

    # 3. PDBFixer
    fixer = PDBFixer(filename=str(raw))
    fixer.applyMutations(["SER-537-TYR"], "A")
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    n_a = len(list(chains[0].residues()))
    fixer.missingResidues = {(0, 0): ["ACE"], (0, n_a): ["NME"]}
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    added = {f"{r.chain.id}:{r.name}{r.id}": [a.name for a in v] for r, v in fixer.missingAtoms.items()}
    fixer.addMissingAtoms(seed=0)
    fixed = WORK / "02_fixed.pdb"
    with open(fixed, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    log["pdbfixer"] = {"mutation": "S537Y (reverts the engineered Y537S to wild type)",
                       "caps_chain_A": "ACE before Leu308, NME after His547",
                       "missing_heavy_atoms_rebuilt": added}

    fx = [a for a in atoms(fixed)]
    rec_a = [a for a in fx if a["chain"] == "A"]
    rec_p = [a for a in fx if a["chain"] != "A" and a["name"] != "OXT"]
    for a in rec_p:
        a["chain"] = pep_chain
    last = max(a["num"] for a in rec_p)
    for a in nh2:
        a.update(res="NHE", num=last + 1, name="N", el="N", rec="ATOM")

    # 4. protonation: pdb2pqr/PROPKA on the uncapped chains, estradiol as ligand
    core = [a for a in rec_a + rec_p if a["res"] not in ("ACE", "NME")]
    core_pdb = WORK / "03_core.pdb"
    write_pdb(core_pdb, [[a for a in core if a["chain"] == "A"], [a for a in core if a["chain"] != "A"]])
    els, pos = ligands.pdb_atoms(REC / "5wgd_rcsb.pdb", resname="EST", chain="A")
    tmpl = ligands.template_from_smiles("C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O")
    est_h, _ = ligands.place_on_pose(tmpl, ligands.heavy_pose(els, pos))
    Chem.MolToMolFile(est_h, str(WORK / "est.sdf"))
    subprocess.run(["antechamber", "-i", "est.sdf", "-fi", "sdf", "-o", "est.mol2", "-fo", "mol2",
                    "-at", "sybyl", "-rn", "EST", "-dr", "no"], cwd=WORK, check=True, capture_output=True)
    cmd = ["pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain", "--titration-state-method=propka",
           f"--with-ph={PH}", "--ligand", "est.mol2", "--pdb-output", "04_protonated.pdb",
           "03_core.pdb", "04_protonated.pqr"]
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    used_ligand = r.returncode == 0
    if not used_ligand:                     # fall back to protein-only optimisation, and say so
        print(r.stderr[-3000:], file=sys.stderr)
        cmd.remove("--ligand")
        cmd.remove("est.mol2")
        subprocess.run(cmd, cwd=WORK, check=True, capture_output=True, text=True)
    states = {}
    for l in open(WORK / "04_protonated.pqr"):
        if l.startswith(("ATOM", "HETATM")):
            f = l.split()
            # pqr: ATOM serial name resname chain resnum ...
            states[(f[4], int(f[5]))] = f[3]
    renamed = {}
    for a in rec_a + rec_p:
        new = states.get((a["chain"], a["num"]))
        std = {"HIS": ("HID", "HIE", "HIP"), "ASP": ("ASH",), "GLU": ("GLH",), "LYS": ("LYN",),
               "CYS": ("CYM", "CYX")}
        if new and a["res"] in std and new in std[a["res"]]:
            renamed[f"{a['chain']}:{a['res']}{a['num']}"] = new
            a["res"] = new
    his = {k: v for k, v in renamed.items() if v.startswith("HI")}
    nonstd = {k: v for k, v in renamed.items() if not v.startswith("HI")}
    o17 = xyz(next(a for a in est if a["name"] == "O17"))
    h524 = {a["name"]: round(float(np.linalg.norm(xyz(a) - o17)), 2) for a in rec_a
            if a["num"] == 524 and a["name"] in ("ND1", "NE2")}
    log["protonation"] = {"method": f"pdb2pqr 3.6 + PROPKA 3.5, pH {PH}, H-bond network optimisation",
                          "estradiol_included_as_ligand": used_ligand,
                          "histidines": his, "other_non_default_states": nonstd or "none",
                          "his524_N_to_estradiol_O17_A": h524,
                          "his524_state": his.get("A:HIS524", renamed.get("A:HIS524"))}
    (REC / "propka").mkdir(exist_ok=True)
    for f in WORK.glob("04_protonated*"):
        if f.suffix in (".pka", ".log"):
            f.replace(REC / "propka" / f.name)

    out = REC / "receptor_wt_src2.pdb"
    write_pdb(out, [rec_a, rec_p + nh2])
    log["output"] = {"file": "receptor/receptor_wt_src2.pdb",
                     "chain_A": f"ACE-{min(a['num'] for a in rec_a if a['res'] != 'ACE')}"
                                f"..{max(a['num'] for a in rec_a if a['res'] != 'NME')}-NME",
                     "heavy_atoms": len(rec_a) + len(rec_p) + len(nh2)}
    yaml.safe_dump(log, open(REC / "receptor_prep.yaml", "w"), sort_keys=False, width=110)
    print(yaml.safe_dump(log, sort_keys=False, width=110))


if __name__ == "__main__":
    main()
