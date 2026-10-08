#!/usr/bin/env bash
# benchmark.sh INPUT_DIR [OUT_TSV]
#
# ns/day of one prepared system (INPUT_DIR holds topol.top conf.gro index.ndx posre_*.itp) under the mdrun settings
# run_replica.sh can use: CUDA graphs on/off x OpenMP threads, all GPU-resident (-update gpu). Minimises briefly,
# heats for 10 ps with the NVT input, then times md.mdp itself (4 fs with HMR, NPT, production output), with the
# counters reset halfway so start-up and PME tuning are excluded. Prints a table and appends it to OUT_TSV (default benchmark.tsv). Takes ~2 min per setting.
set -euo pipefail

IN=$(realpath "$1"); OUT=$(realpath -m "${2:-benchmark.tsv}")
MDP=${ERALPHA_MDP:-/opt/eralpha/mdp}
NSTEPS=${BENCH_NSTEPS:-30000}                         # 120 ps at 4 fs; reset at half
NPROC=$(nproc); MAXT=$(( NPROC < 8 ? NPROC : 8 ))
THREADS=${BENCH_NTOMP:-"$(( MAXT / 2 > 1 ? MAXT / 2 : 1 )) $MAXT"}
WK=$(mktemp -d); trap 'rm -rf "$WK"' EXIT
cd "$WK"; cp -p "$IN"/* .
export GMX_MAXBACKUP=-1

gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo "no GPU")
echo "# $(date -u +%FT%TZ)  $(gmx --version 2>/dev/null | awk -F': *' '/^GROMACS version/ {print $2}')  $gpu  nproc=$NPROC  atoms=$(sed -n 2p conf.gro | tr -d ' ')"

sed 's/@SEED@/1/g' "$MDP/em.mdp" | sed -E 's/^(nsteps *=).*/\1 2000/' > em.mdp
gmx grompp -f em.mdp -c conf.gro -r conf.gro -p topol.top -n index.ndx -o em.tpr -maxwarn 0 > grompp.log 2>&1
gmx mdrun -deffnm em -ntmpi 1 -ntomp "$MAXT" -nb gpu > em.out 2>&1
sed 's/@SEED@/1/g' "$MDP/nvt.mdp" | sed -E 's/^(nsteps *=).*/\1 5000/' > nvt.mdp
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 0 >> grompp.log 2>&1
gmx mdrun -deffnm nvt -ntmpi 1 -ntomp "$MAXT" -nb gpu -pme gpu -bonded gpu -update gpu > nvt.out 2>&1
gmx grompp -f "$MDP/md.mdp" -c nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o bench.tpr -maxwarn 0 \
  >> grompp.log 2>&1

[[ -s $OUT ]] || printf 'date\tgpu\tnproc\tntomp\tcuda_graph\tns_per_day\n' > "$OUT"
printf '%-8s %-11s %s\n' ntomp cuda_graph ns/day
for t in $THREADS; do
  for g in 1 0; do
    rc=0
    GMX_CUDA_GRAPH=$g gmx mdrun -s bench.tpr -deffnm b -ntmpi 1 -ntomp "$t" -pin on \
      -nb gpu -pme gpu -bonded gpu -update gpu -nsteps "$NSTEPS" -resetstep $(( NSTEPS / 2 )) -noconfout \
      > b.out 2>&1 || rc=$?
    perf=$( (( rc == 0 )) && awk '/^Performance:/ {print $2}' b.log || echo "failed($rc)")
    printf '%-8s %-11s %s\n' "$t" "$g" "$perf"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$gpu" "$NPROC" "$t" "$g" "$perf" >> "$OUT"
  done
done
