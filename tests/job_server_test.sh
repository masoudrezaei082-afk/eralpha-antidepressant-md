#!/usr/bin/env bash
# job_server_test.sh [JOB_SERVER]      (inside the production image by .github/workflows/validate.yml; or locally)
#
# docker/job_server.py's contract with Salad's queue worker, with a stub run_replica.sh: /hc answers; exit 0 -> 200
# with progress.json; a non-zero exit -> 500 (a failed attempt, retried); a malformed job -> 400 (ends it); a second
# job while one runs -> 503; SIGTERM reaches run_replica.sh, and the job directory is fresh for every job.
set -uo pipefail
SRV=${1:-/opt/eralpha/bin/job_server.py}
T=$(mktemp -d) PORT=18000
mkdir -p "$T/bin" "$T/work/job"
touch "$T/work/job/leftover"
cat > "$T/bin/run_replica.sh" <<'EOF'
#!/usr/bin/env bash
# stub: SYSTEM decides the outcome
[[ -e leftover ]] && { echo "stale job directory" >&2; exit 9; }
mkdir -p "runs/$1/$2/rep$3/state"
case $1 in
  ok)   echo "{\"status\": \"done\", \"ns\": 90.000, \"seed\": $4}" > "runs/$1/$2/rep$3/state/progress.json"; exit 0 ;;
  fail) exit 3 ;;
  slow) trap 'echo term > "$MARK"; exit 143' TERM; touch "$MARK.started"; while :; do sleep 1; done ;;
esac
EOF
chmod +x "$T/bin/run_replica.sh"
PATH=$T/bin:$PATH WORK=$T/work JOB_PORT=$PORT MARK=$T/mark python3 "$SRV" > "$T/server.log" 2>&1 & pid=$!
req() {   # req METHOD PATH [BODY]: prints the status code, the body goes to $T/body (no curl in the image)
  python3 - "$@" "$PORT" "$T/body" <<'PY'
import sys, urllib.request, urllib.error
m, path, *rest = sys.argv[1:]
body, port, out = (rest[0].encode(), *rest[1:]) if len(rest) == 3 else (None, *rest)
r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=m, headers={"Salad-Job-Id": "t"})
try:
    resp = urllib.request.urlopen(r, timeout=30); code, data = resp.status, resp.read()
except urllib.error.HTTPError as e:
    code, data = e.code, e.read()
except OSError:
    code, data = 0, b""
open(out, "wb").write(data); print(code)
PY
}
post() { req POST / "$1"; }
for _ in $(seq 50); do [[ $(req GET /hc) == 200 ]] && break; sleep 0.2; done
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fi; }
fails=$(
  check "GET /hc -> 200" '[[ $(req GET /hc) == 200 ]]'
  code=$(post '{"system": "ok", "pose": "crystal", "rep": 2, "seed": 77}')
  check "exit 0 -> 200 with progress.json ($code: $(head -c 200 "$T/body"))" \
    '[[ $code == 200 ]] && grep -q "\"ok\": true" "$T/body" && grep -q "\"seed\": 77" "$T/body" && grep -q "\"replica\": \"ok/crystal/2\"" "$T/body"'
  check "job directory removed afterwards" '[[ ! -e $T/work/job ]]'
  code=$(post '{"system": "fail", "pose": "crystal", "rep": 1, "seed": 1}')
  check "exit 3 -> 500 ($code)" '[[ $code == 500 ]] && grep -q "\"exit\": 3" "$T/body"'
  code=$(post '{"system": "../etc", "pose": "x", "rep": 1, "seed": 1}')
  check "malformed job -> 400 ($code)" '[[ $code == 400 ]]'
  code=$(post 'not json')
  check "not JSON -> 400 ($code)" '[[ $code == 400 ]]'
  post '{"system": "slow", "pose": "p", "rep": 1, "seed": 1}' > "$T/slow.code" &
  for _ in $(seq 50); do [[ -e $T/mark.started ]] && break; sleep 0.2; done
  code=$(post '{"system": "ok", "pose": "p", "rep": 1, "seed": 1}')
  check "second job while one runs -> 503 ($code)" '[[ $code == 503 ]]'
) || true
kill -TERM "$pid"; wait "$pid" 2>/dev/null; rc=$?        # the server is this shell's child, not the subshell's
fails+=$'\n'$(check "SIGTERM reaches run_replica.sh; server exits 143 ($rc)" \
  '[[ $(cat "$T/mark" 2>/dev/null) == term && $rc == 143 ]]')
echo "$fails"
grep -q "^FAIL" <<< "$fails" && { echo "--- server log"; cat "$T/server.log"; exit 1; }
exit 0
