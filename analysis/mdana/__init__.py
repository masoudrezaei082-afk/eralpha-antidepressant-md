"""Post-MD analysis of the ERα–SRC2 ligand systems (manuscript 2026MP001112).

Stages (each writes to Drive <eralpha-antidepressant-md>/analysis and skips work already done):
    traj      join parts, complex-only topology/tpr, PBC repair, pocket fit         (traj.process)
    metrics   ligand pose, receptor, coactivator, interactions, QC                 (metrics.run)
    mmgbsa    gmx_MMPBSA GB ε 1/2/4 and PB ε 2, background queue                   (python -m mdana.mmgbsa run)
    aggregate replica-level statistics, tables, clustering                         (aggregate.run)
    figures   main and SI figures                                                  (figures.all_figures)
Long stages run in a background process: python -m mdana.pipeline structure
"""
