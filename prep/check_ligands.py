"""Dry run of the ligand step: template, pose fit, CIP and charge for every system/pose (no files written)."""
import sys
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

sys.path.insert(0, str(Path(__file__).parent))
import ligands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "prep" / "systems.yaml"))
for name, sd in cfg["systems"].items():
    if not sd.get("smiles"):
        continue
    tmpl = (ligands.template_from_allH_file(ROOT / sd["smiles_source"], sd["charge"])
            if sd["smiles"] == "from_pose" else ligands.template_from_smiles(sd["smiles"]))
    print(f"== {name}: {Chem.MolToSmiles(tmpl)}  charge {Chem.GetFormalCharge(tmpl)}  "
          f"{MolToInchiKey(tmpl)}  CIP {ligands._fmt(ligands.cip(tmpl))}")
    for p in sd["poses"]:
        try:
            heavy, info = (ligands.best_glide(tmpl, ROOT, p["cid"]) if p["kind"] == "glide"
                           else ligands.load_pose(p, ROOT, sd))
            mol, rep = ligands.place_on_pose(tmpl, heavy)
            print(f"   {p['id']:9s} {rep['method'][:110]} rmsd={rep['heavy_rmsd_A']} "
                  f"natoms={mol.GetNumAtoms()} q={Chem.GetFormalCharge(mol)} "
                  f"{ {k: v for k, v in info.items() if k in ('docking_score', 'chosen', 'n_candidates', 'glide_microstate_charge')} }")
        except Exception as e:
            print(f"   {p['id']:9s} FAILED: {type(e).__name__}: {e}")
