"""Minimal reader for Schrodinger Maestro (.mae/.maegz) files — enough to pull docked ligand poses
out of a Glide pose-viewer file without a Schrodinger licence.

A Maestro file is a list of `f_m_ct { ... }` blocks. Each has `name` lines, `:::`, then the values,
then indexed sub-blocks such as `m_atom[N] { columns ::: rows ::: }` and `m_bond[N] {...}`.
"""
import gzip
import re

from rdkit import Chem
from rdkit.Geometry import Point3D

# whole-line "# ... #" comments | quoted values (may span lines, e.g. PubChem SD properties) | bare tokens
_TOKEN = re.compile(r'(?m)^[ \t]*(?P<comment>#[^\n]*)$|(?P<tok>"(?:[^"\\]|\\.)*"|\S+)')


def _unquote(tok):
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return tok


def _tokens(text):
    for m in _TOKEN.finditer(text):
        if m.group("tok") is not None:
            yield m.group("tok")


def _parse_block(tok, first):
    """Parse after '{'. Returns (props, subblocks) where subblocks maps name -> list of row dicts."""
    names, props, sub = [], {}, {}
    t = first
    while t != ":::":
        names.append(t)
        t = next(tok)
    for n in names:
        props[n] = _unquote(next(tok))
    t = next(tok)
    while t != "}":
        m = re.match(r"(\w+)\[(\d+)\]$", t)
        if not m:                           # nested non-indexed block, e.g. m_glide_XPvisualizer { }
            assert next(tok) == "{", f"unexpected token {t!r}"
            sub[t] = _parse_block(tok, next(tok))
            t = next(tok)
            continue
        assert next(tok) == "{"
        cols = []
        t = next(tok)
        while t != ":::":
            cols.append(t)
            t = next(tok)
        rows = []
        for _ in range(int(m.group(2))):
            next(tok)                       # row index
            rows.append({c: _unquote(next(tok)) for c in cols})
        assert next(tok) == ":::"
        assert next(tok) == "}"
        sub[m.group(1)] = rows
        t = next(tok)
    return props, sub


def read_cts(path):
    """Yield (properties, atoms, bonds) for every f_m_ct block."""
    opener = gzip.open if str(path).endswith("gz") else open
    with opener(path, "rt") as fh:
        tok = _tokens(fh.read())
    for t in tok:
        if t == "{":                        # file header block
            _parse_block(tok, next(tok))
        elif t == "f_m_ct":
            assert next(tok) == "{"
            props, sub = _parse_block(tok, next(tok))
            yield props, sub.get("m_atom", []), sub.get("m_bond", [])


def ct_to_mol(atoms, bonds):
    """Build an RDKit molecule (all hydrogens, Maestro formal charges, 3D coordinates)."""
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        at = Chem.Atom(int(a["i_m_atomic_number"]))
        at.SetFormalCharge(int(a.get("i_m_formal_charge", 0)))
        at.SetNoImplicit(True)
        rw.AddAtom(at)
        conf.SetAtomPosition(i, Point3D(float(a["r_m_x_coord"]), float(a["r_m_y_coord"]),
                                        float(a["r_m_z_coord"])))
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    seen = set()
    for b in bonds:
        i, j = int(b["i_m_from"]) - 1, int(b["i_m_to"]) - 1
        if (j, i) in seen:                  # Maestro may list each bond in both directions
            continue
        seen.add((i, j))
        rw.AddBond(i, j, order[int(b["i_m_order"])])
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def glide_poses(path):
    """Ligand poses from a Glide *_pv.mae(gz): the first CT is the receptor and is skipped."""
    for k, (props, atoms, bonds) in enumerate(read_cts(path)):
        if k == 0:
            continue
        yield props, ct_to_mol(atoms, bonds)
