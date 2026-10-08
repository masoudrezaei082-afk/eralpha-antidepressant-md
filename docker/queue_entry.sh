#!/usr/bin/env bash
# queue_entry.sh: the image's CMD on Salad. Starts job_server.py (run_replica.sh behind HTTP on :8000), then Salad's
# job queue worker, which takes one job at a time from the group's queue and POSTs it there. If either process
# exits, the other is stopped and the container exits non-zero, so Salad restarts it (restart_policy always).
# SIGTERM goes to both; job_server.py passes it to run_replica.sh, which pushes a checkpoint before it exits.
set -u
job_server.py & srv=$!
for _ in $(seq 60); do                   # the worker should find the server up (the readiness probe also waits)
  (exec 3<>/dev/tcp/127.0.0.1/"${JOB_PORT:-8000}") 2>/dev/null && break
  sleep 1
done
salad-http-job-queue-worker & wrk=$!
stop() { kill -TERM "$wrk" "$srv" 2>/dev/null; }
trap stop TERM INT
wait -n "$srv" "$wrk"
stop
wait "$srv"; rc=$?
wait "$wrk"
exit $(( rc ? rc : 1 ))
