#!/usr/bin/env bash
# run_replica.sh SYSTEM POSE REP SEED
#
# One replica, start to finish, resumable at any point. job_server.py runs it for one Salad queue job in a fresh
# directory (jobs/submit.py queues the jobs). Storage is Google Drive through rclone, at $DRIVE_REMOTE
# (e.g. gdrive:eralpha-antidepressant-md):
#   inputs/<SYSTEM>/<POSE>/                 topol.top conf.gro index.ndx posre_*.itp         pulled at start
#   runs/<SYSTEM>/<POSE>/rep<REP>/state/    only what a resume needs: checkpoints, markers,  pulled at start,
#                                           md.tpr, nodes.txt, perf.tsv, progress.json, DONE  pushed at every checkpoint
#   runs/<SYSTEM>/<POSE>/rep<REP>/data/     trajectory parts, energies, logs                  push only
#   control/prod_ns                         production length in ns, read before every segment (optional)
# Without DRIVE_REMOTE everything stays in ./inputs and ./runs (local tests).
#
# Stages: em -> nvt -> npt_posres_{1000,500,100,10} -> npt -> md. A finished stage leaves state/<stage>.done and
# is never rerun; an interrupted equilibration stage reruns from its start (minutes). Production resumes from its
# last checkpoint (every CPT_MIN minutes), so a preemption costs at most that much GPU time.
#
# mdrun writes in a scratch directory; files are copied into state/ and data/ by an atomic rename. Production runs
# in SEG_H-hour segments with -noappend, so each segment is its own part file. At every new checkpoint data/ is
# pushed before state/, so the checkpoint on Drive never points past the frames on Drive; gmx trjcat drops the
# overlap after a resume. Nothing is ever deleted on Drive (rclone copy, never sync).
#
# Checkpoint safety: a production checkpoint reaches state/ only after gmx dump has read all of it and its step is
# past the one already there; the one it replaces is kept as state/md.prev.cpt. A resume falls back to md.prev.cpt
# when md.cpt does not read, and stops rather than restart production from zero when neither does.
# state/progress.json (stage, secured step and ns, live ns/day, node, GPU load, checkpoint md5) is rewritten at every
# stage and checkpoint; `submit.py status` and `dashboard` read it.
set -euo pipefail

SYSTEM=$1 POSE=$2 REP=$3 SEED=$4
MDP=${ERALPHA_MDP:-/opt/eralpha/mdp}
REL=runs/$SYSTEM/$POSE/rep$REP
IN=$PWD/inputs/$SYSTEM/$POSE
OUT=$PWD/$REL
STATE=$OUT/state DATA=$OUT/data
STAGING=$PWD/.staging/$SYSTEM-$POSE-$REP           # same filesystem as OUT
WK=${SCRATCH:-/tmp}/md-$SYSTEM-$POSE-$REP          # mdrun's working directory
REMOTE=${DRIVE_REMOTE:-}
SEG_H=${SEG_H:-1}
CPT_MIN=${CPT_MIN:-10}
NTOMP=${NTOMP:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}
GPU_FLAGS=${GPU_FLAGS--nb gpu -pme gpu -bonded gpu -update gpu}   # set to "" for a CPU dry run
EM_FLAGS=${EM_FLAGS--nb gpu}
PIN=${PIN:-on}
export GMX_CUDA_GRAPH=${GMX_CUDA_GRAPH:-1}
export GMX_MAXBACKUP=-1                              # never leave #backup# files behind
RC=(rclone --retries 5 --low-level-retries 20 --stats 0 --log-level ERROR)

rm -rf "$WK" "$STAGING"
[[ -n $REMOTE ]] && rm -rf "$OUT"  # Drive is the only truth; leftovers of an earlier job on this node are not
mkdir -p "$IN" "$STATE" "$DATA" "$STAGING" "$WK"
log() { echo "[$(date -u +%FT%TZ)] $SYSTEM/$POSE/rep$REP: $*"; }
die() {       # the reason also goes to state/progress.json on Drive, best effort
  log "ERROR: $*"; trap - TERM INT
  progress failed "$*" 2>/dev/null && [[ -n $REMOTE && -n $PULLED ]] && ! owner_lost \
    && "${RC[@]}" copy "$STATE" "$REMOTE/$REL/state" --include progress.json
  exit 1
}
PULLED= STAGE=start NSTEPS=0 DT=0 CPT_STEP=0 CPT_MD5= CPT_REJECTED=0 SPEED= T_SEG= STEP_SEG= GPU= NODE_START=$(date -u +%FT%TZ)
num() { if [[ $1 =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$1"; else echo null; fi; }
progress() {  # progress STATUS [NOTE] -> state/progress.json, one line of JSON
  local note=${2:-} load gu gt gm ndone ns target
  note=$(tr -d '"\\\n' <<< "$note" | cut -c1-300)
  load=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader,nounits \
           2>/dev/null | head -n 1 | tr -d ' ') || true
  IFS=, read -r gu gt gm <<< "${load:-,,}"
  ndone=$(find "$STATE" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
  ns=$(awk -v s="$CPT_STEP" -v dt="$DT" 'BEGIN {printf "%.3f", s * dt / 1000}')
  target=$(awk -v s="$NSTEPS" -v dt="$DT" 'BEGIN {printf "%.3f", s * dt / 1000}')
  printf '{"utc":"%s","epoch":%s,"status":"%s","note":"%s","stage":"%s","stages_done":%s,"step":%s,"nsteps":%s,' \
    "$(date -u +%FT%TZ)" "$(date +%s)" "$1" "$note" "$STAGE" "$ndone" "$CPT_STEP" "$NSTEPS" > "$STAGING/progress.json"
  printf '"ns":%s,"target_ns":%s,"ns_per_day":%s,"cpt_md5":"%s","cpt_rejected":%s,"machine":"%s","gpu":"%s",' \
    "$ns" "$target" "$(num "$SPEED")" "$CPT_MD5" "$CPT_REJECTED" "${SALAD_MACHINE_ID:-local}" "$GPU" >> "$STAGING/progress.json"
  printf '"gpu_util":%s,"gpu_temp":%s,"gpu_mem_mb":%s,"node_start":"%s"}\n' \
    "$(num "$gu")" "$(num "$gt")" "$(num "$gm")" "$NODE_START" >> "$STAGING/progress.json"
  mv -f "$STAGING/progress.json" "$STATE/progress.json"
}
publish() {   # publish DEST_DIR FILE...  -> atomically into DEST_DIR, skipping files already there unchanged
  local dest=$1 f d; shift
  for f in "$@"; do
    [[ -f $f ]] || continue
    d=$dest/$(basename "$f")
    if [[ -f $d && ! $f -nt $d && $(stat -c %s "$f") == $(stat -c %s "$d") ]]; then continue; fi
    cp -p "$f" "$STAGING/.tmp" && mv -f "$STAGING/.tmp" "$d"
  done
}
OWNER= NEWER=   # this node's line in state/nodes.txt; the last line there
owner_lost() {  # Salad gave this replica to another node (its lease ran out while this one kept running): that node's
                # line follows ours in nodes.txt on Drive. Ours not there yet, or Drive unreadable: keep going.
  [[ -z $REMOTE || -z $OWNER ]] && return 1
  local nodes
  nodes=$("${RC[@]}" cat "$REMOTE/$REL/state/nodes.txt" 2>/dev/null) || return 1
  NEWER=$(tail -n 1 <<< "$nodes")
  grep -qxF -- "$OWNER" <<< "$nodes" && [[ $NEWER != "$OWNER" ]]
}
push() {      # data before state; a failed push is retried at the next checkpoint
  [[ -z $REMOTE ]] && return 0
  if owner_lost; then   # stop without pushing, or the two nodes overwrite each other's parts and checkpoint
    log "another node took this replica over ($(cut -f1,2 <<< "$NEWER")); stopping without pushing"
    trap - TERM INT
    if [[ -n ${pid:-} ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    wait || true
    rm -rf "$WK" "$STAGING" "$OUT"
    exit 3
  fi
  "${RC[@]}" copy "$DATA" "$REMOTE/$REL/data" && "${RC[@]}" copy "$STATE" "$REMOTE/$REL/state" \
    || { log "WARNING: push to $REMOTE failed; retrying at the next checkpoint"; return 1; }
}

if [[ -n $REMOTE ]]; then
  # A failed pull must stop the job: starting from scratch would later overwrite a good checkpoint on Drive.
  "${RC[@]}" copy "$REMOTE/inputs/$SYSTEM/$POSE" "$IN" || die "cannot pull inputs from $REMOTE"
  "${RC[@]}" copy "$REMOTE/$REL/state" "$STATE" || die "cannot pull state from $REMOTE (submit creates it)"
  PULLED=1
fi
if [[ -f $STATE/DONE ]]; then log "already complete"; exit 0; fi
for f in topol.top conf.gro index.ndx; do [[ -s $IN/$f ]] || die "missing input $IN/$f"; done
cp -p "$IN"/* "$WK"/
for f in nodes.txt perf.tsv; do if [[ -f $STATE/$f ]]; then cp -p "$STATE/$f" "$WK"/; fi; done
cd "$WK"
{ printf '%s\t%s\tnproc=%s\tntomp=%s\t' "$(date -u +%FT%TZ)" "${SALAD_MACHINE_ID:-local}" "$(nproc)" "$NTOMP"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
} >> nodes.txt
OWNER=$(tail -n 1 nodes.txt)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1) || GPU=
publish "$STATE" nodes.txt
progress running "node started"
push || true

MDRUN=(gmx mdrun -ntmpi 1 -ntomp "$NTOMP" -pin "$PIN")

grompp() {    # grompp STAGE COORDS [CPT]
  local stage=$1 coords=$2 cpt=${3:-} ref=em.gro
  [[ $stage == em ]] && ref=conf.gro
  sed "s/@SEED@/$SEED/g" "$MDP/$stage.mdp" > "$stage.mdp"
  gmx grompp -f "$stage.mdp" -c "$coords" -r "$ref" ${cpt:+-t "$cpt"} -p topol.top -n index.ndx \
    -o "$stage.tpr" -po "$stage.out.mdp" -maxwarn 0 > "$stage.grompp.log" 2>&1 \
    || { tail -n 30 "$stage.grompp.log"; die "grompp $stage failed"; }
}

# ---------------------------------------------------------------- equilibration
prev_gro=conf.gro prev_cpt=
for stage in em nvt npt_posres_1000 npt_posres_500 npt_posres_100 npt_posres_10 npt; do
  if [[ -f $STATE/$stage.done ]]; then
    cp -p "$STATE/$stage.gro" .
    if [[ -f $STATE/$stage.cpt ]]; then cp -p "$STATE/$stage.cpt" .; fi
  else
    log "stage $stage"
    STAGE=$stage; progress running; push || true
    if [[ $stage == em ]]; then
      grompp em conf.gro
      "${MDRUN[@]}" -deffnm em $EM_FLAGS > em.mdrun.out 2>&1 || { tail -n 30 em.mdrun.out; die "mdrun em failed"; }
      grep -q "converged to Fmax" em.log || die "energy minimisation did not converge"
    else
      [[ $stage == nvt ]] && prev_cpt=     # NVT generates velocities (gen-vel = yes, gen-seed = SEED)
      grompp "$stage" "$prev_gro" "$prev_cpt"
      "${MDRUN[@]}" -deffnm "$stage" $GPU_FLAGS > "$stage.mdrun.out" 2>&1 \
        || { tail -n 30 "$stage.mdrun.out"; die "mdrun $stage failed"; }
    fi
    publish "$DATA" "$stage.edr" "$stage.log" "$stage.out.mdp"
    publish "$STATE" "$stage.gro" "$stage.cpt"
    touch "$stage.done" && publish "$STATE" "$stage.done"
    push || true
  fi
  prev_gro=$stage.gro prev_cpt=$stage.cpt
done

# ---------------------------------------------------------------- production
STAGE=md
tpr_value() { gmx dump -s md.tpr 2>/dev/null | awk -v k="$1" '$1 ~ k && !n {print $NF; n = 1}'; }
if [[ -f $STATE/md.tpr ]]; then cp -p "$STATE/md.tpr" .
else grompp md npt.gro npt.cpt; publish "$DATA" md.out.mdp; publish "$STATE" md.tpr; fi
NSTEPS=$(tpr_value '^nsteps$')
DT=$(tpr_value '^(dt|delta[-_]t)$')
[[ $NSTEPS =~ ^[0-9]+$ ]] || die "cannot read nsteps from md.tpr"
[[ $DT =~ ^[0-9.]+$ ]] || die "cannot read the time step from md.tpr"
cpt_step() {  # cpt_step FILE: its step, or failure unless gmx dump reads the whole file (pipefail)
  local s
  s=$(gmx dump -cp "$1" 2>/dev/null | awk '$1 == "step" && !n {print $NF; n = 1}') || return 1
  [[ $s =~ ^[0-9]+$ ]] && echo "$s"
}

set_length() {  # control/prod_ns on Drive sets the production length of every replica, including running ones
  [[ -z $REMOTE ]] && return 0
  local ns want
  ns=$("${RC[@]}" cat "$REMOTE/control/prod_ns" 2>/dev/null | tr -dc '0-9.') || return 0   # unreadable: keep
  [[ $ns =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 0
  want=$(awk -v ns="$ns" -v dt="$DT" 'BEGIN {printf "%d", ns * 1000 / dt + 0.5}')
  (( want > 0 && want != NSTEPS )) || return 0
  log "production length: $NSTEPS -> $want steps ($ns ns, control/prod_ns)"
  gmx convert-tpr -s md.tpr -nsteps "$want" -o md.new.tpr > convert-tpr.log 2>&1 \
    || { tail -n 20 convert-tpr.log; die "gmx convert-tpr failed"; }
  mv -f md.new.tpr md.tpr && NSTEPS=$want
  publish "$STATE" md.tpr
}

snapshot() {  # parts first, then the checkpoint they cover if it validates; then the same order on Drive
  publish "$DATA" md.part*.xtc md.part*.edr md.part*.log md.part*.trr md.part*.gro
  local c=$STAGING/md.cpt s now
  if [[ -f md.cpt ]] && cp -p md.cpt "$c"; then   # validate a copy: mdrun may replace md.cpt meanwhile
    if s=$(cpt_step "$c") && (( s >= CPT_STEP )); then
      if (( s > CPT_STEP )); then
        if [[ -f $STATE/md.cpt ]]; then cp -p "$STATE/md.cpt" "$STAGING/.tmp" && mv -f "$STAGING/.tmp" "$STATE/md.prev.cpt"; fi
        CPT_MD5=$(md5sum < "$c" | cut -c1-32)
        mv -f "$c" "$STATE/md.cpt"
        now=$(date +%s)
        if [[ -n $T_SEG ]] && (( now > T_SEG && s > STEP_SEG )); then
          SPEED=$(awk -v ds=$(( s - STEP_SEG )) -v dt="$DT" -v t=$(( now - T_SEG )) 'BEGIN {printf "%.1f", ds * dt / 1000 / t * 86400}')
        fi
        T_SEG=$now STEP_SEG=$s CPT_STEP=$s
      fi
    else
      CPT_REJECTED=$(( CPT_REJECTED + 1 ))
      log "WARNING: md.cpt did not validate (step '${s:-unreadable}'); Drive keeps step $CPT_STEP"
    fi
    rm -f "$c"
  fi
  progress running
  push || true
}
pid=
on_term() {   # Salad stops the container: let mdrun write a checkpoint at the next step, then save it
  log "SIGTERM: stopping mdrun at the next checkpoint"
  if [[ -n $pid ]]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  snapshot; progress stopped "SIGTERM"; push || true; exit 143
}
trap on_term TERM INT

for c in md.cpt md.prev.cpt; do    # resume from the newest checkpoint on Drive that reads in full
  [[ -f $STATE/$c ]] || continue
  if s=$(cpt_step "$STATE/$c"); then
    if [[ $c != md.cpt ]]; then
      log "WARNING: resuming from md.prev.cpt (step $s); md.cpt on Drive is replaced by it"
      cp -p "$STATE/$c" "$STAGING/.tmp" && mv -f "$STAGING/.tmp" "$STATE/md.cpt"
    fi
    cp -p "$STATE/md.cpt" md.cpt; CPT_STEP=$s CPT_MD5=$(md5sum < md.cpt | cut -c1-32); break
  fi
  CPT_REJECTED=$(( CPT_REJECTED + 1 )); log "WARNING: $c on Drive does not validate"
done
if [[ ! -f md.cpt ]] && [[ -f $STATE/md.cpt || -f $STATE/md.prev.cpt ]]; then
  die "no production checkpoint on Drive validates (md.cpt, md.prev.cpt); not restarting production from zero"
fi
progress running "production from step $CPT_STEP"; push || true
while :; do
  set_length
  step=0
  if [[ -f md.cpt ]] && ! step=$(cpt_step md.cpt); then   # mdrun's own latest; fall back to the one on Drive
    log "WARNING: local md.cpt does not validate; continuing from step $CPT_STEP"
    CPT_REJECTED=$(( CPT_REJECTED + 1 )); step=$CPT_STEP
    if [[ -f $STATE/md.cpt ]]; then cp -p "$STATE/md.cpt" md.cpt; else rm -f md.cpt; fi
  fi
  (( step >= NSTEPS )) && break
  T_SEG=$(date +%s) STEP_SEG=$step
  log "production from step $step / $NSTEPS (segment ${SEG_H} h, ntomp $NTOMP, CUDA graphs $GMX_CUDA_GRAPH)"
  cpi=(); [[ -f md.cpt ]] && cpi=(-cpi md.cpt)
  "${MDRUN[@]}" -s md.tpr -deffnm md "${cpi[@]}" -noappend -cpt "$CPT_MIN" -maxh "$SEG_H" $GPU_FLAGS \
    > md.mdrun.out 2>&1 &
  pid=$!
  last=$(stat -c %Y md.cpt 2>/dev/null || echo 0)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15 & wait $!                                # interruptible, so SIGTERM is handled at once
    now=$(stat -c %Y md.cpt 2>/dev/null || echo 0)
    if (( now != last )); then sleep 2; snapshot; last=$now; fi
  done
  rc=0; wait "$pid" || rc=$?; pid=
  snapshot
  (( rc == 0 )) || { tail -n 40 md.mdrun.out; die "mdrun md failed (exit $rc)"; }
  perf=$(grep -h "^Performance:" "$(ls md.part*.log | tail -n 1)" | awk '{print $2}' || true)
  if [[ -n $perf ]]; then
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${SALAD_MACHINE_ID:-local}" "$perf" >> perf.tsv
    publish "$STATE" perf.tsv
  fi
done

log "production finished ($NSTEPS steps)"
progress done
date -u +%FT%TZ > DONE && publish "$STATE" DONE
if [[ -n $REMOTE ]]; then      # the job counts as complete only once everything is on Drive
  for try in 1 2 3 4 5; do push && break; (( try == 5 )) && die "final push failed"; sleep 60; done
  rm -rf "$OUT"
fi
rm -rf "$WK" "$STAGING"
