#!/usr/bin/env bash
# run_render.sh: the render image's CMD on a Salad GPU node (jobs/salad_render.py). For each system in ANIM_KEYS
# whose MP4 is not yet on Drive anim-out/: fetch its inputs from Drive anim-inputs/<key>/, render with Cycles on the
# GPU, upload <key>.mp4, <key>_poster.png and <key>_panel.png. A system takes 1.5-2.5 hours, so a preempted
# node that is reallocated resumes at the first system without an MP4 (frames are not kept on Drive). Logs go to
# anim-out/logs/<group>.log.
set -uo pipefail
: "${ANIM_KEYS:?set ANIM_KEYS}" "${DRIVE_REMOTE:?set DRIVE_REMOTE}"
NAME=${SALAD_CONTAINER_GROUP_NAME:-render}
OUT=$DRIVE_REMOTE/anim-out
RC=(rclone --retries 5 --low-level-retries 20 --stats 0 --log-level ERROR)
mkdir -p /tmp/in /tmp/out
LOG=/tmp/out/render.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $NAME: $*"; }
push_log() { "${RC[@]}" copyto "$LOG" "$OUT/logs/$NAME.log" || true; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/GPU: /" || true

for key in $ANIM_KEYS; do
  if "${RC[@]}" lsf "$OUT/$key.mp4" 2>/dev/null | grep -q .; then log "$key: already on Drive"; continue; fi
  "${RC[@]}" copy "$DRIVE_REMOTE/anim-inputs/$key" "/tmp/in/$key" || { log "$key: no inputs"; push_log; continue; }
  log "$key: rendering"
  # a system takes 1.5-2.5 h at 48 samples (the cut-away pocket view is the costly part); the cap stops a hung
  # render from keeping the GPU node billing, and a node far too slow on the overview frames gives up (exit 3)
  timeout "${RENDER_TIMEOUT:-18000}" python -u /opt/eralpha/render.py "/tmp/in/$key" /tmp/out --device GPU \
    --samples "${SAMPLES:-48}" --max-intro-s "${MAX_INTRO_S:-15}" &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 120
    push_log
  done
  wait $pid; rc=$?
  if [ $rc -eq 3 ]; then
    # leave the container so the instance can be reallocated to another node (salad_render.py reallocate)
    log "$key: slow node, giving up"; push_log; exit 3
  fi
  if [ $rc -eq 0 ] && [ -s "/tmp/out/$key.mp4" ]; then
    "${RC[@]}" copy /tmp/out "$OUT" --include "$key.mp4" --include "${key}_poster.png" --include "${key}_panel.png"
    log "$key: uploaded"
  else
    log "$key: render failed (exit $rc)"
  fi
  rm -rf "/tmp/out/${key}_frames"
  push_log
done
log "all done"
push_log
