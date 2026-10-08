#!/usr/bin/env bash
# colab_gpu_test.sh IMAGE INPUT_DIR OUT_DIR       (as root on a Colab GPU VM, e.g. `colab ... --gpu L4`)
#
# Runs the production image without Docker: the image is flattened with crane into a root filesystem and used as a
# chroot (its GROMACS needs Ubuntu 24.04's glibc), with the host's NVIDIA driver libraries bind-mounted in, the same
# way the NVIDIA container runtime does it. INPUT_DIR is one built system (topol.top conf.gro index.ndx posre_*.itp).
#   1. nvidia-smi and `gmx --version` inside the image (CUDA build, SASS for the GPU's compute capability)
#   2. docker/benchmark.sh: ns/day for CUDA graphs on/off x OpenMP threads              -> OUT_DIR/benchmark.tsv
#   3. resume test: run_replica.sh with short stages, SIGKILL during production (a reclaimed node: no signal reaches
#      mdrun), new "node" with only state/ restored, run to the end, check the joined trajectory is continuous
#                                                                                       -> OUT_DIR/resume.txt
# For a private GHCR package, export GHCR_USER and GHCR_TOKEN (read:packages); they are used once for the pull and
# the crane config is deleted afterwards.
set -euo pipefail
IMG=$1 IN=$(realpath "$2") OUT=$(realpath -m "$3")
R=/content/rootfs CRANE=v0.20.6
mkdir -p "$OUT"

if [[ ! -x $R/opt/gromacs/bin/gmx ]]; then
  command -v crane >/dev/null || curl -fsSL \
    "https://github.com/google/go-containerregistry/releases/download/$CRANE/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar xz -C /usr/local/bin crane
  if [[ -n ${GHCR_TOKEN:-} ]]; then echo "$GHCR_TOKEN" | crane auth login ghcr.io -u "$GHCR_USER" --password-stdin; fi
  rm -rf "$R"; mkdir -p "$R"
  crane export --platform linux/amd64 "$IMG" - | tar x -C "$R"
  rm -rf ~/.docker
  crane digest "$IMG" > "$OUT/image_digest.txt" 2>/dev/null || true
fi

drv=$(dirname "$(ldconfig -p | awk '/libcuda.so.1 / {print $NF; exit}')")   # /usr/lib64-nvidia on Colab
mkdir -p "$R/host-nvidia" "$R/dev" "$R/proc" "$R/sys"
mountpoint -q "$R/host-nvidia" || mount --bind "$drv" "$R/host-nvidia"
for d in dev proc sys; do mountpoint -q "$R/$d" || mount --rbind "/$d" "$R/$d"; done
cp -f "$(command -v nvidia-smi)" "$R/usr/local/bin/nvidia-smi"
cp -f /etc/resolv.conf "$R/etc/resolv.conf" 2>/dev/null || true

rm -rf "$R/test"; mkdir -p "$R/test/in"
cp -p "$IN"/topol.top "$IN"/conf.gro "$IN"/index.ndx "$IN"/posre_*.itp "$R/test/in/"
in_image() {
  chroot "$R" env -i HOME=/root PATH=/opt/eralpha/bin:/opt/gromacs/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/host-nvidia GMXLIB=/opt/gromacs/share/gromacs/top ERALPHA_MDP=/opt/eralpha/mdp \
    "$@"
}

# ---------------------------------------------------------------- 1. build check
{ in_image nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  in_image gmx --version 2>/dev/null | grep -E "GROMACS version|GPU support|SIMD|CUDA (compiler|driver|runtime)|sm_"
} | tee "$OUT/gpu_check.txt"

# ---------------------------------------------------------------- 2. benchmark
in_image bash -c 'cd /test && benchmark.sh /test/in /test/benchmark.tsv' | tee "$OUT/benchmark.txt"
cp "$R/test/benchmark.tsv" "$OUT/benchmark.tsv"

# ---------------------------------------------------------------- 3. resume test
set +e                                                      # report every check, even after a failure
T=$R/test/resume; mkdir -p "$T/mdp" "$T/work/inputs/sys"
cp "$R"/opt/eralpha/mdp/*.mdp "$T/mdp/"
sed -i -E 's/^(nsteps *=).*/\1 5000/' "$T/mdp/nvt.mdp"                              # 10 ps at 2 fs
for f in npt_posres_1000 npt_posres_500 npt_posres_100 npt_posres_10 npt; do
  sed -i -E 's/^(nsteps *=).*/\1 2500/' "$T/mdp/$f.mdp"                           # 10 ps each at 4 fs (HMR)
done
sed -i -E 's/^(nsteps *=).*/\1 125000/; s/^(nstxout *=).*/\1 12500/' "$T/mdp/md.mdp"  # 500 ps; trr every 50 ps
cp -p "$R"/test/in/* "$T/work/inputs/sys/"
run() {   # run ATTEMPT -> background run_replica.sh with 1-min checkpoints and 2-min segments
  in_image env ERALPHA_MDP=/test/resume/mdp CPT_MIN=1 SEG_H=0.034 SCRATCH=/test/resume/scratch \
    bash -c "cd /test/resume/work && exec run_replica.sh sys bench 1 4242" > "$T/attempt$1.log" 2>&1
}
STATE=$T/work/runs/sys/bench/rep1/state DATA=$T/work/runs/sys/bench/rep1/data
run 1 & p=$!
until [[ -f $STATE/md.cpt ]] || ! kill -0 $p 2>/dev/null; do sleep 5; done
sleep 75                                                    # past at least one more checkpoint
pkill -KILL -f "run_replica.sh sys bench" || true; pkill -KILL -f "gmx mdrun" || true; wait $p || true
step1=$(in_image gmx dump -cp /test/resume/work/runs/sys/bench/rep1/state/md.cpt 2>/dev/null | awk '$1=="step" {print $NF; exit}')
mv "$DATA" "$T/data_attempt1"                            # data/ is upload-only: the new node starts without it
rm -rf "$T/scratch" "$T/work/.staging"                      # a new node: only state/ (from Drive) survives
run 2; rc2=$?                                               # foreground: runs to DONE
mkdir -p "$T/joined"; cp -p "$T"/data_attempt1/* "$DATA"/* "$T/joined/"
{ echo "attempt 1 killed (SIGKILL) with checkpoint at step $step1"
  echo "attempt 2 exit $rc2"
  grep -h "production from step" "$T"/attempt*.log
  ls -1 "$T/joined" | grep "^md.part"
  in_image bash -c 'cd /test/resume/joined && gmx trjcat -f md.part*.xtc -o all.xtc > trjcat.log 2>&1 && gmx check -f all.xtc 2>&1' \
    | grep -E "^(Step|Coords|Last frame|Timestep)"
  echo "expected: 51 frames, last frame at 500 ps, one timestep of 10 ps"
  ls "$STATE"/DONE
} 2>&1 | tee "$OUT/resume.txt"
grep -h "Performance:" "$T"/joined/md.part*.log | tee -a "$OUT/resume.txt"
