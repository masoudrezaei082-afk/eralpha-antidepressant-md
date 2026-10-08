#!/usr/bin/env bash
# drive_resume_test.sh   (no GPU, no GROMACS, no network: runs anywhere with bash and rclone, in about 2 minutes)
#
# Tests docker/run_replica.sh's storage logic against a local directory standing in for Google Drive (rclone
# treats a plain path as a remote), with a stub `gmx` that writes one line per 1000 steps into each part file.
#   1. node 1 runs equilibration and production, and is SIGKILLed (a reclaimed node: no signal reaches it)
#      once a production checkpoint is on "Drive"
#   2. control/prod_ns is raised while the replica is down
#   3. node 2 starts with an empty /work, pulls state/, resumes and runs to the new length
#   4. checks: DONE and the new length on Drive, every step covered by the part files, the checkpoint on Drive
#      never ahead of the frames on Drive, the node's local copy removed, and a failed pull refusing to start
#   5. takeover (replica 2): node A keeps running while node B is given the same replica (Salad re-leases a job
#      whose node lost its connection); A must stop at its next push without pushing, B finishes the replica
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'pkill -KILL -f "$T/" 2>/dev/null || true; rm -rf "$T"' EXIT
D=$T/drive; mkdir -p "$T/bin" "$D/inputs/sys/bench" "$D/runs/sys/bench/rep1/state" "$D/runs/sys/bench/rep1/data" \
  "$D/control"
printf 'stub\n' | tee "$D/inputs/sys/bench/"{topol.top,conf.gro,index.ndx} > /dev/null
echo 0.16 > "$D/control/prod_ns"                      # 40,000 steps at 4 fs

cat > "$T/bin/gmx" <<'STUB'
#!/usr/bin/env bash
# stub gmx: grompp/mdrun/dump/convert-tpr with text "tpr" and "cpt" files
set -euo pipefail
cmd=$1; shift
arg() { local k=$1; shift; while (( $# )); do [[ $1 == "$k" ]] && { echo "$2"; return; }; shift; done; }
val() { awk -v k="$1" '$1 == k {print $3; exit}' "$2"; }
case $cmd in
  grompp)
    f=$(arg -f "$@"); o=$(arg -o "$@"); po=$(arg -po "$@")
    ns=$(awk '$1 == "nsteps" {print $3}' "$f"); dt=$(awk '$1 == "dt" {print $3}' "$f")
    printf 'nsteps = %s\ndt = %s\n' "$ns" "${dt:-0}" > "$o"; cp "$f" "$po" ;;
  dump)
    if [[ $1 == -s ]]; then cat "$2"; else cat "$2"; fi ;;
  convert-tpr)
    s=$(arg -s "$@"); o=$(arg -o "$@"); n=$(arg -nsteps "$@")
    sed "s/^nsteps = .*/nsteps = $n/" "$s" > "$o" ;;
  mdrun)
    name=$(arg -deffnm "$@")
    if [[ $name != md ]]; then
      for x in gro cpt edr; do echo "$name" > "$name.$x"; done
      echo "converged to Fmax" > "$name.log"; exit 0
    fi
    nsteps=$(val nsteps md.tpr); cpi=$(arg -cpi "$@"); cpt_s=$(( $(arg -cpt "$@") * 2 ))
    maxs=$(awk -v h="$(arg -maxh "$@")" 'BEGIN {printf "%d", h * 3600}')
    step=0 part=1
    if [[ -n $cpi ]]; then step=$(val step "$cpi"); part=$(( $(val part "$cpi") + 1 )); fi
    pf=$(printf 'md.part%04d' "$part"); : > "$pf.xtc"; echo "part $part" > "$pf.log"
    save() { printf 'step = %s\npart = %s\n' "$step" "$part" > md.cpt.tmp; mv md.cpt.tmp md.cpt; }
    trap 'save; exit 0' TERM
    t0=$SECONDS last=$SECONDS
    while (( step < nsteps && SECONDS - t0 < maxs )); do
      sleep 0.5; step=$(( step + 1000 )); echo "$step" >> "$pf.xtc"
      if (( SECONDS - last >= cpt_s )); then save; last=$SECONDS; fi
    done
    save; echo "Performance:      123.4        0.194" >> "$pf.log" ;;
  *) echo "stub gmx: $cmd not stubbed" >&2; exit 1 ;;
esac
STUB
chmod +x "$T/bin/gmx"

run() {   # run NODE [REP]: a fresh /work per node, as on a new Salad machine
  mkdir -p "$T/$1"
  ( cd "$T/$1" && exec env PATH="$T/bin:$PATH" ERALPHA_MDP="$REPO/mdp" SCRATCH="$T/$1/tmp" DRIVE_REMOTE="$D" \
      CPT_MIN=1 SEG_H=0.002 GPU_FLAGS= EM_FLAGS= PIN=off NTOMP=1 \
      bash "$REPO/docker/run_replica.sh" sys bench "${2:-1}" 4242 )
}
fail=0; check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
S=$D/runs/sys/bench/rep1/state P=$D/runs/sys/bench/rep1/data

setsid bash -c "$(declare -f run); T=$T D=$D REPO=$REPO; run node1" > "$T/node1.log" 2>&1 & pg=$!
until [[ -f $S/md.cpt ]] || ! kill -0 $pg 2>/dev/null; do sleep 1; done
kill -KILL -- -$pg 2>/dev/null || true; wait $pg 2>/dev/null || true
step1=$(awk '$1 == "step" {print $3}' "$S/md.cpt")
check "node 1 left a production checkpoint on Drive (step $step1)" '(( step1 > 0 ))'
check "checkpoint on Drive is covered by the frames on Drive" \
  '(( $(cat "$P"/md.part*.xtc | sort -n | tail -n 1) >= step1 ))'

echo 0.2 > "$D/control/prod_ns"                       # 50,000 steps
run node2 > "$T/node2.log" 2>&1 && rc=0 || rc=$?
check "node 2 exited 0" '(( rc == 0 ))'
check "node 2 resumed from step $step1" 'grep -q "production from step $step1 /" "$T/node2.log"'
check "node 2 applied control/prod_ns (50000 steps)" 'grep -q -- "-> 50000 steps" "$T/node2.log"'
check "DONE and md.tpr with the new length on Drive" '[[ -f $S/DONE ]] && grep -q "nsteps = 50000" "$S/md.tpr"'
check "every 1000-step frame from 1000 to 50000 is on Drive" \
  'diff <(cat "$P"/md.part*.xtc | sort -nu) <(seq 1000 1000 50000) > /dev/null'
check "equilibration markers and logs on Drive" '[[ -f $S/npt.done && -f $P/npt.log && -f $P/md.out.mdp ]]'
check "both nodes logged in nodes.txt" '(( $(wc -l < "$S/nodes.txt") == 2 ))'
check "node 2 removed its local copy" '[[ ! -d $T/node2/runs/sys/bench/rep1 ]]'
mv "$S" "$S.hidden"
run node3 > "$T/node3.log" 2>&1 && rc=0 || rc=$?
check "a failed state pull stops the job (exit $rc) before any work" \
  '(( rc != 0 )) && grep -q "cannot pull state" "$T/node3.log" && [[ ! -e $S ]]'
S2=$D/runs/sys/bench/rep2/state P2=$D/runs/sys/bench/rep2/data; mkdir -p "$S2"
setsid bash -c "$(declare -f run); T=$T D=$D REPO=$REPO; run nodeA 2" > "$T/nodeA.log" 2>&1 & pa=$!
until [[ -f $S2/md.cpt ]] || ! kill -0 $pa 2>/dev/null; do sleep 1; done
run nodeB 2 > "$T/nodeB.log" 2>&1 && rc=0 || rc=$?
wait $pa && rca=0 || rca=$?
check "node B finished the replica (exit $rc)" '(( rc == 0 )) && [[ -f $S2/DONE ]]'
check "node A stopped without pushing once node B registered (exit $rca)" \
  '(( rca == 3 )) && grep -q "another node took this replica over" "$T/nodeA.log"'
check "node A stopped before node B finished" '! grep -q "production finished" "$T/nodeA.log"'
check "every 1000-step frame from 1000 to 50000 is on Drive (replica 2)" \
  'diff <(cat "$P2"/md.part*.xtc | sort -nu) <(seq 1000 1000 50000) > /dev/null'
check "node A removed its local copy" '[[ ! -d $T/nodeA/runs/sys/bench/rep2 ]]'
(( fail == 0 )) || { for n in node1 node2 nodeA nodeB; do echo "--- $n"; cat "$T/$n.log"; done; exit 1; }
echo "all checks passed"
