"""Where everything lives, which systems exist, and how residues are numbered.

Drive layout written by the production run (docs/cloud_run.md):
    <DRIVE>/inputs/<sys>/<pose>/{topol.top,index.ndx,conf.gro}
    <DRIVE>/runs/<sys>/<pose>/rep<k>/{data/md.partNNNN.{xtc,edr,log}, state/{md.tpr,DONE}}
Analysis outputs go to <DRIVE>/analysis/ (never into runs/; branches/ is never read).
"""
import os
from dataclasses import dataclass
from pathlib import Path

DRIVE = Path(os.environ.get("MDANA_DRIVE", "/content/drive/MyDrive/eralpha-antidepressant-md"))
WORK = Path(os.environ.get("MDANA_WORK", "/content/work"))          # fast local scratch on the VM
OUT = DRIVE / "analysis"
REPO = Path(__file__).resolve().parents[2]

REPLICAS = (1, 2, 3)
PROD_NS = 100.0
DT_PS = 10.0                       # xtc output interval (Complex atoms only)
N_FRAMES = int(PROD_NS * 1000 / DT_PS) + 1   # frame 0 at t = 0 -> 10,001 frames
T_EQ_NS = 20.0                     # first 20 ns of production are excluded from averages (docs/MD_protocol.md, Simulation)
CONTACT_STRIDE = 5                 # contacts / H-bonds every 50 ps

# Receptor residues in the built systems are numbered 1..242 (ACE, Leu308..His547, NME); crystal = gro + 306.
# The SRC2 peptide follows: 243 ACE, 244-254 HKILHKLLQDS, 255 NHE.
CRYSTAL_OFFSET = 306
REC_RESID = (2, 241)               # Leu308..His547
PEP_RESID = (244, 254)
PEP_SEQ = "HKILHKLLQDS"
LOOP_CRYSTAL = (461, 472)          # modelled loop (grafted from SWISS-MODEL), excluded from the receptor fit
H12_CRYSTAL = (536, 544)
ANCHORS = {"Glu353": 353, "Arg394": 394, "His524": 524}
LIGAND_RESNAME = "LIG"


def gro_resid(crystal_resnum: int) -> int:
    return crystal_resnum - CRYSTAL_OFFSET


def label_resid(resid: int, resname: str) -> str:
    """Human-readable residue label: crystal numbering for ERα, position in the peptide for SRC2."""
    three = resname[:1] + resname[1:3].lower()
    three = {"Hid": "His", "Hie": "His", "Hip": "His", "Cyx": "Cys", "Ash": "Asp", "Glh": "Glu", "Lyn": "Lys"}.get(
        three, three)
    if REC_RESID[0] <= resid <= REC_RESID[1]:
        return f"{three}{resid + CRYSTAL_OFFSET}"
    if PEP_RESID[0] <= resid <= PEP_RESID[1]:
        return f"SRC2-{three}{resid - PEP_RESID[0] + 1}"
    return f"{three}{resid}"


@dataclass(frozen=True)
class System:
    key: str
    pose: str
    label: str
    color: str
    ligand: bool = True
    linestyle: str = "-"

    def rep_id(self, rep: int) -> str:
        return f"{self.key}_rep{rep}"

    def run_dir(self, rep: int) -> Path:
        return DRIVE / "runs" / self.key / self.pose / f"rep{rep}"

    def input_dir(self) -> Path:
        return DRIVE / "inputs" / self.key / self.pose


# Okabe–Ito colours (colour-blind safe), fixed per system in every figure. The two fluoxetine enantiomers share a
# purple hue; the negative control is grey and dashed, apo light grey.
SYSTEMS = [
    System("estradiol", "crystal", "Estradiol", "#000000"),
    System("pape1", "autodock", "PaPE-1", "#0072B2"),
    System("imipramine", "glide", "Imipramine", "#D55E00"),
    System("s_ketamine", "glide", "S-Ketamine", "#009E73"),
    System("hnk", "glide", "(2R,6R)-HNK", "#E69F00"),
    System("r_fluoxetine", "glide", "R-Fluoxetine", "#CC79A7"),
    System("s_fluoxetine", "glide", "S-Fluoxetine", "#882E72"),
    System("decoy", "biovia", "Decoy", "#7F7F7F", linestyle="--"),
    System("apo", "apo", "Apo", "#BBBBBB", ligand=False),
]
BY_KEY = {s.key: s for s in SYSTEMS}
LIGANDED = [s for s in SYSTEMS if s.ligand]
REFERENCE = "estradiol"


def replicas(keys=None):
    """(System, rep) for every replica, optionally restricted to system keys or 'key_repN' ids."""
    out = []
    for s in SYSTEMS:
        for r in REPLICAS:
            if keys and s.key not in keys and s.rep_id(r) not in keys:
                continue
            out.append((s, r))
    return out


def proc_dir(s: System, rep: int) -> Path:
    """Per-replica analysis outputs on Drive (processed trajectory, metrics, done markers)."""
    return OUT / "replicas" / s.rep_id(rep)


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
