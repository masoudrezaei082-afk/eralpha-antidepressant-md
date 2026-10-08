# MD protocol — ERα LBD with antidepressants

Every setting below is built by a script (`prep/`), recorded per system in `systems/<name>/provenance.yaml`, and cited
from `docs/references.bib` (DOIs verified against Crossref with `bibverify`).

## 1. Receptor (`prep/build_receptor.py` → `receptor/receptor_prep.yaml`)

- **Source:** PDB 5WGD chain A, ERα LBD + estradiol + SRC2 coactivator peptide, 1.80 Å [@speltz2018_5wgd]. Altloc A
  is used. Crystal waters are removed.
- **Wild type:** the crystallised construct carries the constitutively active Y537S mutation, which stabilises the
  agonist AF-2 conformation [@fanning2016y537s]. It is reverted (S537Y, PDBFixer) because the experiments use
  wild-type ERα.
- **Coactivator:** the SRC2 peptide in contact with chain A's AF-2 groove is kept (chain E, `ACE-HKILHKLLQDS-NH2`,
  43 heavy-atom contacts; chain F has none). This holds helix 12 in the agonist position estradiol was crystallised
  in [@brzozowski1997er].
- **Missing loop 461–472:** grafted from a SWISS-MODEL model built on the same template [@waterhouse2018swissmodel]
  (Cα RMSD 0.08 Å to 5WGD; junction C–N 1.31/1.50 Å, relaxed during minimisation).
- **Missing side-chain atoms** (REMARK 470: 11 residues of chain A, Asp11 of the peptide) and the reverted Tyr537 side chain
  are rebuilt with PDBFixer [@eastman2024openmm8].
- **Termini:** chain A is capped with ACE (before Leu308) and NME (after His547), because residues 294–307 and
  548–554 are not resolved and free charged termini there would be artificial. The peptide keeps its crystal
  acetyl and C-terminal amide (AMBER NHE).
- **Protonation at pH 7.4:** PDB2PQR + PROPKA3, with the H-bond network optimised in the presence of crystal
  estradiol [@jurrus2018apbs; @olsson2011propka3]. No Asp/Glu/Lys/Cys is in a non-default state.
  - His524 is assigned **HIE**: Nδ1 is 2.82 Å from the estradiol 17β-O and accepts that hydroxyl's hydrogen bond.
  - All 14 His states are listed in `receptor_prep.yaml`.
- **The same receptor file is used for every system**, including apo. Ligands differ only in the ligand.

## 2. Ligands (`prep/ligands.py`, `prep/systems.yaml`)

- **Microstate at pH 7.4:**
  - Imipramine (tertiary amine, pKa ≈ 9.5), fluoxetine (≈ 9.8), S-ketamine (≈ 7.5) and (2R,6R)-HNK are simulated
    **protonated (+1)**.
  - Estradiol and PaPE-1 are neutral.
  - The decoy is +1 on its piperidine N, as supplied.
  - The previous runs used neutral free bases throughout.
- **Stereochemistry is enforced, not inherited from docking.**
  - Each ligand's template comes from an explicit-stereo SMILES (PubChem CID in `systems.yaml`). Each docked pose is
    mapped onto it by graph matching.
  - RDKit CIP labels of the pose are compared with the intended labels:
    - **Match:** the pose coordinates are kept and only H atoms are added.
    - **Mismatch:** the intended stereoisomer is rebuilt into the pose. One refit moves only the inverted
      substituent (all other heavy atoms pinned, MMFF-relaxed); the other is the best of 300 ETKDGv3 conformers
      aligned to the pose. Whichever is closer to the pose is used. Heavy-atom RMSD to the pose is logged.

| System | Pose | Docked stereo | Action | Heavy-atom RMSD to pose |
|---|---|---|---|---|
| S-fluoxetine | Glide XP (XP_1, racemic CID 3386) | R | branch refit to S | 0.61 Å |
| (2R,6R)-HNK | Glide XP (XP_1, CID 137795569) | as intended (2R,6R) | coordinates kept | 0 |
| all others | — | as intended | coordinates kept | 0 |

`provenance.yaml` keys CIP labels by RDKit atom index (HNK `2R,5R` = atoms 2 and 5, i.e. IUPAC C2 and C6).

- **Parameters:** GAFF2 atom types with AM1-BCC charges (antechamber/sqm) [@wang2004gaff; @jakalian2002am1bcc].
  - Charges are computed **once per ligand** and reused for both poses and all replicas. Their rounding residual
    (≤ 0.002 e) is spread evenly, so the net charge is exactly integral.
  - Missing terms come from parmchk2 (none flagged "ATTN, need revision" so far).
  - MD and MM/GBSA use the **same** topology, so the earlier ff99SB/GAFF-vs-GAFF2 mismatch cannot recur.
- **Starting poses: Schrödinger Glide XP** [@friesner2006glidexp], the docking the manuscript reports.
  - Every Glide XP run delivered (`glide-dock_XP_1`, `_2`, `_3`, `_5_IFDPrep`) is pooled. All four used the same
    rigid receptor (5WGD-mod, `glide-grid_1`; receptor coordinates identical to 0.000 Å), so their scores are
    comparable.
  - For each ligand the best-scoring pose with the intended stereochemistry is taken. Only if no run has one is
    the pose needing the smallest refit used (S-fluoxetine).
  - All 3 replicas of an antidepressant start from that pose, each with its own velocity seed.
  - Estradiol starts from the crystal pose (3 replicas). Its Glide pose is built only as a redocking check and not
    run: one replica from it would add no statistics.
  - **PaPE-1 is the exception:** it was never docked with Glide, so its 3 replicas start from the AutoDock4 rank-1
    pose [@morris2009autodock4].
  - The decoy starts from the supplied BIOVIA Discovery Studio pose (3).
  - Each pose's source file, docking score, method and RMSD are in `provenance.yaml`.
- **Identities resolved from the files rather than assumed:**
  - **PaPE-1** = `Cc1cc(-c2ccc3c(c2)CC[C@@H]3O)cc(C)c1O` (C17H18O2, 1S-indanol), perceived from the supplied
    all-H structure.
  - **The decoy** = PubChem CID 44748165, N-(3,3-dimethylbutan-2-yl)-2-[4-methoxy-3-(trifluoromethyl)benzoyl]-
    2,8-diazaspiro[4.5]decane-4-carboxamide. It is **record 231 of the official DUD-E ESR1 decoy set**
    [@mysinger2012dude] (ZINC39684576; `decoys_final.sdf.gz`, sha256 `de7c74313a3247dc264f…`). Structure,
    stereochemistry and protonation are identical to the supplied pose (same InChIKey,
    YYCKZXFMNKYSSA-QAPCUYQASA-O). DUD-E decoys are property-matched to ESR1 actives but topologically
    dissimilar, and are *putative* non-binders; the decoy is therefore a negative control, not a measured
    non-binder.

## 3. System (`prep/build_system.py`)

| Setting | Value | Reason |
|---|---|---|
| Protein / water / ions | ff14SB / TIP3P / Joung–Cheatham | current AMBER protein FF; ions fitted to TIP3P [@maier2015ff14sb; @jorgensen1983tip3p; @joung2008ions] |
| Box | truncated octahedron, ≥ 12 Å solute–edge (a = 96.9 Å, ~700 nm³) | about 23 % smaller than a cube of the same image distance; 12 Å avoids self-interaction of the AF-2/peptide region |
| Salt | neutralised (6 Na⁺), then 0.15 M NaCl (54 pairs from N_water/55.5 M) | physiological ionic strength, the same in every system and in MM/GBSA |
| Size | ~63,500 atoms (19,750 waters) | measured |
| Position restraints | heavy atoms of protein, peptide and ligand; `-DPOSRES_FC=` sets the force constant per stage | one topology for all stages |
| Index groups | `Receptor`, `Ligand`, `Complex`, `Solvent_Ions` written by the script | no interactive `make_ndx` |
| Check | `grompp -maxwarn 0` for EM and NVT; net charge 0 | a build fails rather than warns |

## 4. Simulation (`mdp/`, GROMACS 2026)

Shared by all stages:
- Verlet buffer, PME (0.12 nm grid, order 4) [@essmann1995pme], 1.0 nm real-space and LJ cutoff with potential
  shift, and the dispersion correction for energy and pressure (the AMBER-consistent scheme).
- H-bond constraints with LINCS [@hess2008plincs].
- **Hydrogen mass repartitioning** (`mass-repartition-factor = 3`: H = 3.024 u, the mass taken from the bonded heavy
  atom, water included) with a **4 fs** time step [@hopkins2015hmr]. NVT heating keeps 2 fs. HMR leaves equilibrium
  properties unchanged but can slightly change the rates of dynamics, which matter only for kinetics, not claimed
  here.
  It halves the cost of the sampling, which is what made 3 replicas per system affordable.
- 310 K: cells and brain at 37 °C.
- V-rescale thermostat, τ_T = 1 ps [@bussi2007vrescale].
- C-rescale barostat at 1 bar, τ_P = 2 ps, in every NPT stage [@bernetti2020crescale]. It gives correct volume
  fluctuations and is safe from the start of equilibration, unlike Parrinello–Rahman.

| Stage | Length | Restraint (kJ mol⁻¹ nm⁻²) | Notes |
|---|---|---|---|
| EM | ≤ 50,000 steps, Fmax < 500 | 1000 | steepest descent |
| NVT | 500 ps | 1000 | velocities drawn with the replica seed |
| NPT | 4 × 500 ps | 1000 → 500 → 100 → 10 | stepwise release; `refcoord-scaling = com` |
| NPT | 2 ns | none | unrestrained equilibration |
| Production | **100 ns × 3 replicas** | none | Complex every 10 ps (xtc), full system every 2 ns, energies every 10 ps, checkpoint every 10 min |

- **Total sampling:** 9 systems × 3 replicas × 94.5 ns ≈ 2.6 µs.
- The production length is set once for all replicas (`control/prod_ns`, `docs/cloud_run.md`) from the measured
  speed and the budget; 90 ns was the plan, extended to 100 ns (`length 100`), and the Methods report what was run.
- The first 20 ns of each production replica are treated as further equilibration. They are excluded from
  averages, and that cut is checked by block analysis.

## 5. Analysis plan (`analysis/`)

Per replica, then across replicas (n = 3; mean ± SEM, t-based 95 % CI, t₂ = 4.30, so only large differences are resolved and
the text says so):
- **Pose stability:** ligand heavy-atom RMSD after fitting on pocket Cα (within 6 Å of the starting ligand).
  **Pose retention** is the fraction of frames under 2.0 Å.
- Ligand–pocket centroid distance.
- Receptor backbone RMSD and per-residue RMSF; helix-12 position relative to the SRC2 groove.
- **Contacts and hydrogen bonds:** occupancy of the canonical ERα anchor residues Glu353, Arg394 and His524, plus
  every residue in contact > 20 % of the time [@brzozowski1997er].
- **Convergence:** cumulative averages and block averaging [@flyvbjerg1989blocking]. Replica-to-replica spread is
  shown, not hidden.
- **MM/GBSA** (gmx_MMPBSA [@valdes2021gmxmmpbsa], `analysis/mmpbsa_gb.in`):
  - Single-trajectory protocol, 160 frames per replica (one per 0.5 ns, 20–100 ns), 480 per system.
  - GB-neck2 with mbondi3 radii, 0.15 M salt, at ε_in = 1, 2, 4. Per-residue decomposition within 6 Å.
  - PB at ε_in = 2 as a cross-check (`mmpbsa_pb.in`). No entropy term.
  - Reported as a ranking relative to estradiol with SEM across replicas. A difference between two ligands is
    discussed only when it holds at every ε_in and its 95 % CI excludes zero.

## 6. Software

GROMACS 2026 [@abraham2015gromacs]; AmberTools (tleap, antechamber, sqm, parmchk2) [@case2023ambertools]; ParmEd
[@shirts2017parmed]; RDKit 2026.03; OpenMM/PDBFixer 8.6 [@eastman2024openmm8]; PDB2PQR/PROPKA3; gmx_MMPBSA 1.7.
Exact versions are pinned in `docker/tools-env.yml` and the GPU image tag.
