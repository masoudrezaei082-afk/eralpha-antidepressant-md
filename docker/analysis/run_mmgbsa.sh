#!/usr/bin/env bash
# run_mmgbsa.sh: the CPU analysis image's CMD on Salad. One container group per ligand system (MM_KEYS = system
# keys or key_repN ids), one instance each, created by jobs/salad_mmgbsa.py.
#
# The rclone remote has drive.file scope, so it sees only what rclone wrote (inputs/, runs/), not the analysis/
# folder the Colab notebook writes through the Drive mount. Each replica is therefore processed here with the same
# code (mdana.traj.process: join parts, complex topology/tpr, PBC repair, pocket fit) from runs/ and inputs/, then
# its MM/GBSA jobs run one by one with mpirun over all vCPUs (python -m mdana.mmgbsa run --only <tag>). Each
# finished job goes to Drive <DRIVE_REMOTE>/mmgbsa-salad/<id>/<tag>/ at once (the notebook reads it via the mount).
# A job already DONE there is skipped, so a preempted node that is reallocated resumes where it stopped.
# A failed job leaves its gmx_MMPBSA log there as failed_run.log; the script's log goes to
# mmgbsa-salad/salad_logs/<group>.log after every job. Nothing on Drive is deleted; runs/ and inputs/ are read only.
set -uo pipefail
: "${MM_KEYS:?set MM_KEYS}" "${DRIVE_REMOTE:?set DRIVE_REMOTE}"
NP=${NP:-$(nproc)}
TAGS=${TAGS:-gb_eps2 gb_eps1 gb_eps4 pb_eps2}
DECOMP_EPS=${DECOMP_EPS:-2}
NAME=${SALAD_CONTAINER_GROUP_NAME:-${MM_KEYS// /_}}
LOCAL=$MDANA_DRIVE/analysis
LOG=$MDANA_WORK/run_mmgbsa.log
RC=(rclone --retries 5 --low-level-retries 20 --stats 0 --log-level ERROR)
mkdir -p "$MDANA_WORK" "$LOCAL"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $NAME: $*"; }
push_log() { "${RC[@]}" copyto "$LOG" "$DRIVE_REMOTE/mmgbsa-salad/salad_logs/$NAME.log" || true; }

RESULTS=$DRIVE_REMOTE/mmgbsa-salad
reps=$(python -c "
import os
from mdana import layout as L
print(' '.join(f'{s.rep_id(r)}:{s.key}:{s.pose}:{r}' for s, r in L.replicas(os.environ['MM_KEYS'].split()) if s.ligand))")
log "replicas: $reps; tags: $TAGS; $NP MPI ranks on $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM"
failed=0
for item in $reps; do
  IFS=: read -r id sys pose rep <<< "$item"
  rel=runs/$sys/$pose/rep$rep
  t0=$(date +%s)
  # what is already finished on Drive (DONE markers; the decomposition table decides whether gb_eps2 is complete)
  "${RC[@]}" copy "$RESULTS/$id" "$LOCAL/mmgbsa/$id" --include "*/DONE" --include "*/FINAL_DECOMP_MMPBSA.csv" \
    2>/dev/null || true
  todo=
  for tag in $TAGS; do
    [[ -f $LOCAL/mmgbsa/$id/$tag/DONE && ( $tag != gb_eps"$DECOMP_EPS" || -z $DECOMP_EPS
       || -f $LOCAL/mmgbsa/$id/$tag/FINAL_DECOMP_MMPBSA.csv ) ]] || todo+=" $tag"
  done
  [[ -z $todo ]] && { log "$id: all jobs already DONE on Drive"; continue; }
  "${RC[@]}" copy "$DRIVE_REMOTE/inputs/$sys/$pose" "$MDANA_DRIVE/inputs/$sys/$pose" \
    --include topol.top --include conf.gro --include index.ndx \
  && "${RC[@]}" copy "$DRIVE_REMOTE/$rel/state" "$MDANA_DRIVE/$rel/state" --include md.tpr --include DONE \
  && "${RC[@]}" copy "$DRIVE_REMOTE/$rel/data" "$MDANA_DRIVE/$rel/data" --include "md.part*.xtc" \
    --include "md.part*.edr" --transfers 8 \
  && python -c "
from mdana import layout as L, traj
c = traj.process(L.BY_KEY['$sys'], $rep)
print('$id processed:', {k: c[k] for k in ('n_frames', 'max_min_dist_ligand_protein_A', 'parts')})" \
    || { log "$id: trajectory processing failed"; failed=1; push_log; continue; }
  rm -rf "$MDANA_DRIVE/$rel/data"
  log "$id: trajectory processed in $(( ($(date +%s) - t0) / 60 )) min"
  for tag in $TAGS; do
    t0=$(date +%s)
    python -m mdana.mmgbsa run --keys "$id" --only "$tag" --decomp-eps $DECOMP_EPS --np "$NP"
    out=$LOCAL/mmgbsa/$id/$tag
    if [[ -f $out/DONE ]]; then
      "${RC[@]}" copy "$out" "$RESULTS/$id/$tag" || { log "$id/$tag: push failed"; failed=1; }
      log "$id/$tag: done in $(( ($(date +%s) - t0) / 60 )) min"
    else
      wl=$MDANA_WORK/mmgbsa/$id/$tag/run.log
      [[ -f $wl ]] && "${RC[@]}" copyto "$wl" "$RESULTS/$id/$tag/failed_run.log"
      log "$id/$tag: FAILED"; failed=1
    fi
    push_log
  done
  rm -rf "$LOCAL/replicas/$id" "$MDANA_WORK/mmgbsa/$id" "$MDANA_WORK/$id"
done
log "finished (failed=$failed)"
push_log
exit $failed
