#!/usr/bin/env bash
# ci_replica_test.sh SYSTEM POSE INPUT_DIR OUT_DIR      (inside the production image, CPU only; .github/workflows/validate.yml)
#
# The production image's own run_replica.sh, on CPU, on one real built system, with every stage shortened and a
# local folder standing in for Google Drive (the image's rclone treats a plain path as a remote):
#   em -> nvt (2 fs) -> npt_posres_{1000,500,100,10} -> npt (4 fs, HMR) -> md, then
#   1. node 1 is SIGKILLed (a reclaimed node: no signal reaches it) once its second production checkpoint is on
#      "Drive", so state/ holds md.cpt and md.prev.cpt
#   2. md.cpt on "Drive" is truncated to half (a corrupt checkpoint), and control/prod_ns is raised from 0.06 to
#      0.07 ns while the replica is down
#   3. node 2 starts with an empty working directory, pulls state/, rejects md.cpt, resumes from md.prev.cpt and runs
#      to the new length
# Checks: every stage done, DONE and the new nsteps on "Drive", the joined trajectory continuous (71 frames, 0-70 ps),
# HMR masses in md.tpr, no LINCS/SETTLE warnings, mean temperature near 310 K, the checkpoint fallback, and
# state/progress.json (done, 0.07 ns, md5 of the final md.cpt).                          -> OUT_DIR/summary.txt
set -uo pipefail
SYS=$1 POSE=$2 IN=$(realpath "$3") OUT=$(realpath -m "$4")
T=$(mktemp -d) D=$T/drive REL=runs/$SYS/$POSE/rep1
S=$D/$REL/state P=$D/$REL/data
mkdir -p "$OUT" "$T/mdp" "$D/inputs/$SYS/$POSE" "$S" "$P" "$D/control"
cp -p "$IN"/topol.top "$IN"/conf.gro "$IN"/index.ndx "$IN"/posre_*.itp "$D/inputs/$SYS/$POSE/"

cp /opt/eralpha/mdp/*.mdp "$T/mdp/"
set_mdp() { local f=$1; shift; while (( $# )); do sed -i -E "s/^($1 *=).*/\1 $2/" "$T/mdp/$f.mdp"; shift 2; done; }
set_mdp nvt nsteps 1000                                                   # 2 ps at 2 fs
for f in npt_posres_1000 npt_posres_500 npt_posres_100 npt_posres_10 npt; do set_mdp "$f" nsteps 500; done  # 2 ps
set_mdp md nsteps 15000 nstxout-compressed 250 nstenergy 500 nstlog 2500 nstxout 2500   # 60 ps; xtc every 1 ps
echo 0.06 > "$D/control/prod_ns"

run() {   # run NODE: a fresh working directory per node, as on a new Salad machine
  mkdir -p "$T/$1"
  ( cd "$T/$1" && exec env ERALPHA_MDP="$T/mdp" SCRATCH="$T/$1/tmp" DRIVE_REMOTE="$D" CPT_MIN=1 SEG_H=0.03 \
      GPU_FLAGS= EM_FLAGS= PIN=off NTOMP="$(nproc)" run_replica.sh "$SYS" "$POSE" 1 4242 )
}
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fi; }
t0=$SECONDS

setsid bash -c "$(declare -f run); T=$T D=$D SYS=$SYS POSE=$POSE; run node1" > "$OUT/node1.log" 2>&1 & pg=$!
until [[ -f $S/md.prev.cpt ]] || ! kill -0 $pg 2>/dev/null; do sleep 5; done
kill -KILL -- -$pg 2>/dev/null; wait $pg 2>/dev/null
cstep() { gmx dump -cp "$1" 2>/dev/null | awk '$1 == "step" {print $NF; exit}'; }
step1=$(cstep "$S/md.cpt") prev1=$(cstep "$S/md.prev.cpt")
jget() { grep -oE "\"$1\":(\"[^\"]*\"|[^,}]*)" "$2" 2>/dev/null | head -n 1 | cut -d: -f2- | tr -d '"'; }  # no python here
status1=$(jget status "$S/progress.json")
size=$(stat -c %s "$S/md.cpt"); head -c $(( size / 2 )) "$S/md.cpt" > "$T/half" && mv "$T/half" "$S/md.cpt"
gmx dump -cp "$S/md.cpt" > /dev/null 2>&1 && corrupt_reads=yes || corrupt_reads=no
echo 0.07 > "$D/control/prod_ns"                                           # 17,500 steps
run node2 > "$OUT/node2.log" 2>&1; rc=$?

J=$T/joined; mkdir -p "$J"; cp -p "$P"/md.part*.xtc "$P"/md.part*.edr "$J/" 2>/dev/null
( cd "$J" && gmx trjcat -f md.part*.xtc -o all.xtc > trjcat.log 2>&1 && gmx check -f all.xtc > check.log 2>&1
  gmx eneconv -f md.part*.edr -o all.edr > eneconv.log 2>&1
  echo Temperature | gmx energy -f all.edr -o T.xvg > energy.log 2>&1 )
frames=$(awk '$1 == "Coords" {print $2}' "$J/check.log")
last=$(tr '\r' '\n' < "$J/check.log" | grep -oE "Last frame +[0-9]+ +time +[0-9.]+" | awk '{print $NF}')  # on a progress line
tavg=$(awk '/^Temperature/ {print $2}' "$J/energy.log")
gmx dump -s "$S/md.tpr" > "$T/tpr.txt" 2>/dev/null
nsteps=$(awk '$1 == "nsteps" {print $3; exit}' "$T/tpr.txt")
heavyH=$(grep -c 'm= 3.02400e+00' "$T/tpr.txt")
warn=$(cat "$P"/*.log 2>/dev/null | grep -ciE "LINCS WARNING|could not settle|Too many LINCS")
fstep=$(cstep "$S/md.cpt") fprev=$(cstep "$S/md.prev.cpt") fmd5=$(md5sum < "$S/md.cpt" | cut -c1-32)
prog="$(jget status "$S/progress.json") $(jget ns "$S/progress.json") $(jget cpt_md5 "$S/progress.json")"
missing=$(for s in em nvt npt_posres_1000 npt_posres_500 npt_posres_100 npt_posres_10 npt; do
  [[ -f $S/$s.done ]] || printf '%s ' "$s"; done)

{ echo "$SYS/$POSE: CPU, $(nproc) threads, $(( (SECONDS - t0) / 60 )) min"
  check "node 1 left md.cpt (step ${step1:-none}) and md.prev.cpt (step ${prev1:-none}) on Drive" \
    '[[ ${prev1:-0} -gt 0 && ${step1:-0} -gt ${prev1:-0} ]]'
  check "node 1 had not finished before the kill; its progress.json says '$status1'" \
    '! grep -q "production finished" "$OUT/node1.log" && [[ $status1 == running ]]'
  check "gmx dump rejects the half-truncated md.cpt (reads: $corrupt_reads)" '[[ $corrupt_reads == no ]]'
  check "every equilibration stage done${missing:+ (missing: $missing)}" '[[ -z $missing ]]'
  check "node 2 exited 0" '(( rc == 0 ))'
  check "node 2 fell back to md.prev.cpt and resumed from step $prev1" \
    'grep -q "resuming from md.prev.cpt (step $prev1)" "$OUT/node2.log" && grep -q "production from step $prev1 /" "$OUT/node2.log"'
  check "node 2 applied control/prod_ns: md.tpr nsteps $nsteps (want 17500)" '[[ $nsteps == 17500 && -f $S/DONE ]]'
  check "final md.cpt on Drive reads, step $fstep (want 17500); md.prev.cpt reads, step $fprev" \
    '[[ $fstep == 17500 && ${fprev:-0} -gt 0 && $fprev -lt 17500 ]]'
  check "progress.json: $prog (want: done 0.07 <md5 of md.cpt>)" '[[ $prog == "done 0.070 $fmd5" ]]'
  check "joined trajectory continuous: $frames frames, last at $last ps (want 71, 70)" \
    '[[ $frames == 71 ]] && awk -v t="$last" "BEGIN {exit !(t > 69.9 && t < 70.1)}"'
  check "HMR: $heavyH atoms with mass 3.024 u in md.tpr" '(( heavyH > 1000 ))'
  check "no LINCS/SETTLE warnings in any log ($warn)" '(( warn == 0 ))'
  check "mean production temperature $tavg K (want 300-320)" \
    'awk -v t="${tavg:-0}" "BEGIN {exit !(t > 300 && t < 320)}"'
  echo "CPU speed (perf.tsv, ns/day): $(cut -f3 "$S/perf.tsv" 2>/dev/null | paste -sd ' ')"
  echo "progress.json: $(cat "$S/progress.json" 2>/dev/null)"
} | tee "$OUT/summary.txt"
cp -p "$P"/*.log "$S/nodes.txt" "$S/progress.json" "$J/check.log" "$OUT/" 2>/dev/null
grep -q "^FAIL" "$OUT/summary.txt" && exit 1
exit 0
