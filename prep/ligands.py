"""Ligand handling: read a docked pose, put the intended pH-7.4 microstate and stereoisomer onto it.

Every pose source (AutoDock PDB with polar H only, Glide pose with LigPrep's own protonation, crystal
HETATM, BIOVIA PDB) is reduced to heavy-atom coordinates. Bond orders, formal charges and stereo
come only from the template (systems.yaml SMILES). The pose therefore cannot smuggle in a wrong
protonation or enantiomer.

If the pose has the wrong configuration at a stereocentre (the old HNK pose is (2R,6S); docking
programs often return the other fluoxetine or ketamine enantiomer), the correct stereoisomer is
embedded (ETKDGv3, 300 conformers). The conformer closest to the pose is kept, over all symmetry-
equivalent atom mappings, and its heavy-atom RMSD to the pose is recorded in provenance.yaml.
"""
import itertools
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler, rdDetermineBonds, rdMolAlign
from rdkit.Geometry import Point3D

from maestro import glide_poses

N_CONFS = 300
SEED = 20261006


# ---------------------------------------------------------------- reading coordinates
def _element(line):
    """Element symbol: last column if it is one (AutoDock-derived files shift it), else cols 77-78,
    else the first letter of the atom name."""
    line = line.rstrip()
    last = line.split()[-1]
    if len(line) > 66 and re.fullmatch(r"[A-Za-z]{1,2}[0-9]?[+-]?", last):
        el = re.sub(r"[^A-Za-z]", "", last)
    else:
        el = re.sub(r"[^A-Za-z]", "", line[76:78]) or re.sub(r"[^A-Za-z]", "", line[12:16])[:1]
    return el.capitalize()


def pdb_atoms(path, resname=None, chain=None, first_model=True):
    """(elements, coords) from ATOM/HETATM records; altloc A or blank; optional residue/chain filter."""
    els, xyz = [], []
    for line in open(path):
        if line.startswith("ENDMDL") and first_model and els:
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16] in "BCDEFGH" and line[17:20].strip():   # alternate locations other than A
            continue
        if resname and line[17:20].strip() != resname:
            continue
        if chain and line[21] != chain:
            continue
        els.append(_element(line))
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return els, np.array(xyz)


def mol_from_xyz(els, xyz, charge=0, bond_orders=False):
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(els))
    for i, (e, p) in enumerate(zip(els, xyz)):
        rw.AddAtom(Chem.Atom(e))
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    if bond_orders:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    else:
        rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def heavy_pose(els, xyz):
    keep = [i for i, e in enumerate(els) if e != "H"]
    return mol_from_xyz([els[i] for i in keep], xyz[keep])


# ---------------------------------------------------------------- templates and stereo
def template_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def template_from_allH_file(path, charge):
    """Perceive a molecule (bond orders, charges, stereo from 3D) from a pose that has every H."""
    els, xyz = pdb_atoms(path)
    mol = mol_from_xyz(els, xyz, charge=charge, bond_orders=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return Chem.RemoveHs(mol)


def cip(mol):
    """{atom index: 'R'/'S'} for tetrahedral centres (new CIP labeller)."""
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}


def _onto_template(tmpl, pose):
    """Heavy-atom pose renumbered into template order, with the template's bond orders and charges."""
    match = _topology_match(tmpl, pose)
    if not match:
        raise ValueError("pose connectivity does not match the template")
    rw = Chem.RWMol(tmpl)
    rw.RemoveAllConformers()
    conf = Chem.Conformer(tmpl.GetNumAtoms())
    pc = pose.GetConformer()
    for ti, pi in enumerate(match):
        conf.SetAtomPosition(ti, pc.GetAtomPosition(pi))
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    return m


def _topology_match(tmpl, pose):
    """Graph match on elements + connectivity only (bond orders, aromaticity and charges ignored)."""
    q = Chem.RWMol(tmpl)
    Chem.Kekulize(q, clearAromaticFlags=True)
    for a in q.GetAtoms():
        a.SetFormalCharge(0)
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    qq = Chem.AdjustQueryProperties(q.GetMol(), params)
    p = Chem.Mol(pose)
    p.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(p)
    return p.GetSubstructMatch(qq)


def _automorphisms(tmpl):
    q = Chem.Mol(tmpl)
    return q.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=5000)


def place_on_pose(tmpl, pose):
    """Return (mol with H in template order, report). `pose` = heavy-atom mol with 3D coordinates."""
    want = cip(tmpl)
    on = _onto_template(tmpl, pose)
    got = cip(on)
    if got == want:
        mh = Chem.AddHs(on, addCoords=True)
        return mh, {"method": "pose coordinates kept; H added", "heavy_rmsd_A": 0.0,
                    "pose_cip": _fmt(got)}

    # Wrong configuration at >= 1 centre. Two refits; the one closer to the pose wins:
    #  (a) branch: re-embed with every heavy atom pinned to the pose except one acyclic substituent
    #      of each inverted centre, so only that substituent moves to the other face;
    #  (b) global: best of N_CONFS ETKDGv3/MMFF conformers aligned to the pose.
    bad = [i for i in want if got.get(i) != want[i]]
    tries = [_branch_refit(tmpl, on, bad, want), _global_refit(tmpl, on)]
    rmsd, mol, how = min((t for t in tries if t), key=lambda t: t[0])
    return mol, {"method": f"pose had {_fmt(got)}, intended {_fmt(want)}: {how}",
                 "heavy_rmsd_A": round(float(rmsd), 3), "pose_cip": _fmt(got)}


def _heavy_rmsd(mol, ref, idx):
    a, b = mol.GetConformer().GetPositions()[idx], ref.GetConformer().GetPositions()[idx]
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _side(mol, centre, start):
    """Atoms reachable from `start` without passing through `centre`."""
    seen, stack = {start}, [start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j != centre and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _branch_refit(tmpl, on, bad, want):
    heavy = list(range(tmpl.GetNumAtoms()))
    options = []
    for c in bad:
        opts = [(_side(tmpl, c, nb.GetIdx()) | {c}) for nb in tmpl.GetAtomWithIdx(c).GetNeighbors()
                if not tmpl.GetBondBetweenAtoms(c, nb.GetIdx()).IsInRing()]
        options.append(opts)
    if not all(options):
        return None
    pc = on.GetConformer()
    best = None
    for combo in itertools.product(*options):
        free = set().union(*combo)
        if len(free) > len(heavy) // 2:      # moving most of the molecule is not a local fix
            continue
        pinned = [i for i in heavy if i not in free]
        cmap = {i: pc.GetAtomPosition(i) for i in pinned}
        for k in range(10):
            mh = Chem.AddHs(tmpl)
            if AllChem.EmbedMolecule(mh, coordMap=cmap, randomSeed=SEED + k, useRandomCoords=True) < 0:
                continue
            props = AllChem.MMFFGetMoleculeProperties(mh)
            ff = AllChem.MMFFGetMoleculeForceField(mh, props)
            rdMolAlign.AlignMol(mh, on, atomMap=[(i, i) for i in pinned])
            for i in pinned:
                ff.MMFFAddPositionConstraint(i, 0.05, 100.0)
            ff.Minimize(maxIts=2000)
            rdMolAlign.AlignMol(mh, on, atomMap=[(i, i) for i in pinned])
            Chem.AssignStereochemistryFrom3D(mh)
            if cip(mh) != want:
                continue
            r = _heavy_rmsd(mh, on, heavy)
            if best is None or r < best[0]:
                moved = sorted(free - set(bad))
                best = (r, Chem.Mol(mh), f"substituent atoms {moved} re-embedded on the other face, "
                                         f"all other heavy atoms pinned to the pose (MMFF-relaxed)")
    return best


def _global_refit(tmpl, on):
    mh = Chem.AddHs(tmpl)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=N_CONFS, params=ps))
    AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=500)
    heavy = [a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() > 1]
    best = (1e9, None, None)
    for perm in _automorphisms(tmpl):
        amap = [(i, perm[i]) for i in heavy]
        for c in cids:
            r = rdMolAlign.AlignMol(mh, on, prbCid=c, atomMap=amap)
            if r < best[0]:
                best = (r, c, amap)
    rmsd, c, amap = best
    rdMolAlign.AlignMol(mh, on, prbCid=c, atomMap=amap)
    out = _single_conf(mh, c)
    Chem.AssignStereochemistryFrom3D(out)
    return rmsd, out, f"best of {N_CONFS} ETKDGv3/MMFF conformers of the intended stereoisomer, aligned to the pose"


def _single_conf(mol, cid):
    m = Chem.Mol(mol)
    keep = m.GetConformer(cid)
    conf = Chem.Conformer(keep)
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    return m


def cip_text(mol):
    return _fmt(cip(mol))


def _fmt(labels):
    return ",".join(f"{i}{v}" for i, v in sorted(labels.items())) or "achiral"


# ---------------------------------------------------------------- pose sources
def load_pose(spec, root, sysdef):
    """Heavy-atom pose mol for one systems.yaml pose entry, plus a description of its source."""
    kind = spec["kind"]
    if kind == "autodock":
        path = root / "systems" / spec["dir"] / "source" / "autodock_rank1.pdb"
        return heavy_pose(*pdb_atoms(path)), {"source": str(path.relative_to(root)),
                                              "program": "AutoDock4 rank-1 (model 1)"}
    if kind == "crystal":
        path = root / "receptor" / "5wgd_rcsb.pdb"
        return heavy_pose(*pdb_atoms(path, resname=spec["resname"], chain="A")), \
            {"source": f"{path.relative_to(root)} chain A {spec['resname']}", "program": "X-ray (PDB 5WGD)"}
    if kind == "file":
        path = root / spec["path"]
        return heavy_pose(*pdb_atoms(path)), {"source": spec["path"], "program": "BIOVIA Discovery Studio"}
    raise ValueError(kind)


def glide_candidates(root, cid):
    """All Glide XP poses whose title is this PubChem CID, as (docking score, heavy pose, info)."""
    out = []
    for f in sorted((root / "systems" / "_glide_xp").glob("*_pv.maegz")):
        for props, mol in glide_poses(f):
            if props.get("s_m_title") != str(cid):
                continue
            heavy = Chem.RemoveHs(mol)
            info = {"source": f"{f.relative_to(root)} (title {cid})", "program": "Glide XP",
                    "docking_score": float(props["r_i_docking_score"]),
                    "glide_microstate_charge": Chem.GetFormalCharge(mol)}
            out.append((info["docking_score"], heavy, info))
    return sorted(out, key=lambda t: t[0])


def best_glide(tmpl, root, cid):
    """Best-scoring Glide pose that already has the intended stereo; else the best-scoring pose (refit)."""
    cands = glide_candidates(root, cid)
    if not cands:
        raise FileNotFoundError(f"no Glide pose titled {cid}")
    want = cip(tmpl)
    for score, heavy, info in cands:
        if cip(_onto_template(tmpl, heavy)) == want:
            return heavy, {**info, "chosen": "best-scoring pose with the intended stereochemistry",
                           "n_candidates": len(cands)}
    # none has the intended stereo: take the pose that needs the smallest refit
    fits = [(place_on_pose(tmpl, heavy)[1]["heavy_rmsd_A"], score, heavy, info) for score, heavy, info in cands]
    rmsd, score, heavy, info = min(fits, key=lambda t: (t[0], t[1]))
    return heavy, {**info, "chosen": "no candidate had the intended stereo; pose needing the smallest refit",
                   "candidate_stereo": [cip_text(_onto_template(tmpl, h)) for _, h, _ in cands],
                   "n_candidates": len(cands)}
