# eralpha-antidepressant-md

Replicated all-atom molecular dynamics of the human estrogen receptor α (ERα) ligand-binding domain bound to the
SRC2 coactivator peptide, with antidepressants, 17β-estradiol, the membrane-ERα agonist PaPE-1 and a decoy.

| | |
|---|---|
| Receptor | ERα LBD + SRC2 peptide, PDB 5WGD (wild type; missing loop grafted from SWISS-MODEL) |
| Systems (9) | apo · 17β-estradiol (positive control) · PaPE-1 · imipramine · (S)-ketamine · (R)-fluoxetine · (S)-fluoxetine · (2R,6R)-hydroxynorketamine · DUD-E ESR1 decoy (negative control, PubChem CID 44748165) |
| Sampling | 3 independent replicas × 100 ns per system (27 runs, 2.7 µs); analysis window 20–100 ns |
| Force field | ff14SB · GAFF2/AM1-BCC · TIP3P · Joung–Cheatham ions, 0.15 M NaCl |
| Engine | GROMACS 2026.4 (CUDA), 4 fs with hydrogen mass repartitioning, 310 K, 1 bar (C-rescale) |
| End-point energies | gmx_MMPBSA 1.7.0: GB-neck2 at ε_in = 1, 2, 4 and PB at ε_in = 2. These are end-point energies without entropy, reported as a ranking against estradiol, not as affinities |
| Trajectories | Zenodo, [doi:10.5281/zenodo.23237415](https://doi.org/10.5281/zenodo.23237415) |

The full protocol, with a citation for every setting, is in [`docs/MD_protocol.md`](docs/MD_protocol.md). The Methods
text generated from the built systems is in [`docs/methods_text.md`](docs/methods_text.md).

## What is where

```
receptor/        5WGD input, the prepared receptor (receptor_wt_src2.pdb) and receptor_prep.yaml (every decision)
systems/         _glide_xp/  Glide XP pose-viewer files (the docked starting poses)
                 <system>/   provenance.yaml, GAFF2 parameters (params/), starting pose and ligand files
prep/            systems.yaml (the single list of systems, SMILES, charges, poses, replicas) and the build scripts
mdp/             em → nvt → npt_posres_{1000,500,100,10} → npt → md (100 ns)
analysis/        mdana/ (the analysis package), gmx_MMPBSA inputs (mmpbsa_gb.in, mmpbsa_pb.in), anim/ (renders)
analysis/results figures/ (PDF, SVG, 600-dpi PNG) and tables/ (CSV, XLSX, summary.json): the published numbers
notebooks/       md_prep.ipynb            rebuilds every system from the inputs (CPU, Colab)
                 md_analysis.ipynb        all analysis, statistics, energies and figures (CPU, Colab)
                 md_analysis_executed.ipynb  the executed run with every figure inline (paths as run)
docker/          GPU production image, CPU tools image, analysis and render images; pinned tools-env.lock.txt
jobs/            replica manifest (seeds) and the SaladCloud submission scripts used for production
docs/            protocol, Methods text, references.bib, cloud_run.md
tests/           CI and resume tests for the production container
```

Figures in `analysis/results/figures/`:

| File | Content |
|---|---|
| `Fig_MD_rmsd` | ligand heavy-atom RMSD (symmetry-corrected) to the starting pose, every replica plus mean ± SEM |
| `Fig_MD_interactions` | (a) pose retention, (b) ΔΔE versus estradiol at every ε_in and PB, (c) contact and H-bond occupancy |
| `FigS1a`, `FigS1b` | receptor backbone and helix-12 RMSD |
| `FigS2` | per-residue Cα RMSF against apo |
| `FigS3` | SRC2 peptide RMSD and charge-clamp distances |
| `FigS4a`, `FigS4b` | ligand–pocket centroid distance and ligand–protein minimum distance |
| `FigS5` | convergence: running means, half-split comparison, block SEM |
| `FigS6` | per-residue MM/GBSA decomposition (ε_in = 2) |
| `FigS7` | sensitivity to ε_in and GB against PB |
| `FigS8` | run QC: temperature, pressure, density, energy and volume |

The main summary table is `analysis/results/tables/main_table.csv`.

## Reproduce

Each step can start from the previous step's published output, so you can begin wherever you like.

The Zenodo record has the same folder layout that the notebooks read from Google Drive:
`inputs/` (built systems), `runs/` (run inputs and energies) and `analysis/` (processed trajectories, per-replica
metrics and gmx_MMPBSA outputs).

### 1. Re-draw the figures and tables (minutes)

1. Download and unpack the Zenodo record.
2. Copy it to your Google Drive, open [`notebooks/md_analysis.ipynb`](notebooks/md_analysis.ipynb) in Colab (a free
   CPU runtime is enough), set `DRIVE_ROOT` to that folder, and run all cells.

Every stage whose output already exists is skipped, so only the statistics, tables and figures are recomputed.

### 2. Re-run the analysis (hours, CPU)

Delete the stage outputs you want recomputed under `analysis/`, or call the stage with `force=True`, and run the
notebook again. MM/GBSA is the slow part: 72 GB jobs and 24 PB jobs, which run in a background process
(`python -m mdana.mmgbsa`). Running `mdana` outside the notebook reads the folder from the environment variable
`MDANA_DRIVE`.

Tools environment (GROMACS 2026.3 CPU, AmberTools, MDAnalysis), pinned:

```bash
micromamba create -p ./env --file docker/tools-env.lock.txt && micromamba activate ./env
pip install gmx_MMPBSA==1.7.0 spyrmsd
```

### 3. Rebuild the simulation systems (CPU)

Open [`notebooks/md_prep.ipynb`](notebooks/md_prep.ipynb) in Colab and run all cells, or, in the tools environment:

```bash
python -I prep/build_receptor.py      # receptor/receptor_wt_src2.pdb + receptor_prep.yaml
python -I prep/check_ligands.py       # dry run: microstate, stereochemistry and pose fit per ligand
python -I prep/build_system.py        # all systems, or name some: build_system.py estradiol apo
```

`build_system.py` fails unless every system passes `gmx grompp -maxwarn 0` with net charge 0. The built
topologies (`systems/<system>/<pose>/build/`) are not committed, because the script regenerates them. They are also
in the Zenodo record (`inputs/`).

### 4. Re-run production (GPU)

Any GROMACS ≥ 2026 build can run one replica from a built system:

```bash
docker/run_replica.sh SYSTEM POSE REP SEED      # one row of jobs/manifest.yaml
```

This runs em → NVT → stepwise-restrained NPT → NPT → 100 ns production, resuming from the last checkpoint after
an interruption. `jobs/manifest.yaml` lists the fixed seed of every replica, so re-running a replica regenerates
the same initial velocities.

The production runs used SaladCloud RTX 4070 Ti GPUs with the image from `docker/Dockerfile` and Google Drive as
storage. The setup is in [`docs/cloud_run.md`](docs/cloud_run.md). Credentials go in a local `.env` (see
`.env.example`) and never in the image or in git.

## Animations

Rendered videos of every ligand system are attached to the
[latest release](../../releases/latest).

## License

Code: MIT ([`LICENSE`](LICENSE)). Data, figures and tables: CC BY 4.0. Third-party structures keep their own
terms: PDB 5WGD, and the DUD-E decoy set.
