# Running production on SaladCloud + Google Drive

All 27 replicas (9 systems × 3) run as jobs of a **Salad job queue** (`eralpha-antidepressant-md`) on a Salad container group
(`eralpha-antidepressant-md-q`) of RTX 4070 Ti / 4070 Ti Super GPUs at medium priority, one replica per GPU. Inputs, checkpoints and
results live in one folder on the Google Drive of moleculardynamicsbio1@gmail.com, reached through rclone.
`jobs/submit.py` does every step and takes `--dry-run` before any command (`python jobs/submit.py --dry-run submit`).

One job is one whole replica (≈ 6 h). In each container, Salad's queue worker takes one job at a time and POSTs it
to `job_server.py` on port 8000, which runs `run_replica.sh` and answers when it exits: 200 once the replica is
`DONE` on Drive, 500 if it failed (`docker/queue_entry.sh` starts both).

## Storage: what is written, and how much

Per replica (63.5k atoms, 4,145-atom `Complex` group, 90 ns, `mdp/md.mdp`):

| Output | Frequency | Size |
|---|---|---|
| `md.part*.xtc`, Complex only | 10 ps (9,000 frames) | ≈ 0.17 GB |
| `md.part*.trr`, full system, positions only | 2 ns (45 frames) | ≈ 0.04 GB |
| `md.part*.edr` / `.log` | 10 ps / 100 ps | ≈ 0.01 GB |
| `state/` (latest `md.cpt`, stage `.gro`/`.cpt`, `md.tpr`) | overwritten in place | ≈ 0.02 GB |
| equilibration `edr`/`log`/`mdp` | once | ≈ 0.01 GB |
| **Total per replica** | | **≈ 0.25 GB** |

27 replicas ≈ **7 GB**, plus inputs (≈ 0.1 GB) and MM/GBSA outputs later (≈ 1 GB). The account has 5 TB, so storage
costs nothing. Drive keeps older revisions of overwritten files (the growing part file, `md.cpt`) for 30 days; at
this size that is a few GB at most, and they expire on their own.

## Privacy

- rclone uses **your own OAuth client** with the **`drive.file`** scope: it can see and change only files it created
  itself. Nothing else on that Drive is visible to it, or to anything holding its token.
- The folder is not shared. Files are private to moleculardynamicsbio1 unless you share them yourself.
- The token reaches Salad only as container-group environment variables, which Salad stores encrypted. It is never in
  the image, the job definitions or git. Revoke it at any time under Google Account → Security → Third-party access.

## Layout and sync design

```
gdrive:eralpha-antidepressant-md/
  inputs/<system>/<pose>/                topol.top conf.gro index.ndx posre_*.itp        written once by `inputs`
  runs/<system>/<pose>/rep<k>/state/     *.done, stage .gro/.cpt, md.tpr, md.cpt,      pulled on resume
                                         md.prev.cpt, nodes.txt, perf.tsv,
                                         progress.json, DONE
  runs/<system>/<pose>/rep<k>/data/      md.partNNNN.{xtc,trr,edr,log,gro},           pushed only
                                         equilibration edr/log/mdp
  control/prod_ns                        production length in ns, for every replica    set by `length`
```

The queue only hands out jobs. `run_replica.sh` moves the files itself with rclone:
- **On start** it pulls `inputs/` and its own `state/`. A failed pull stops the job before any work, so a node that
  cannot see Drive never restarts a replica from zero.
- **Checkpoint every 10 min** (`CPT_MIN`). At each new checkpoint it pushes `data/` first and then `state/`, so the
  checkpoint on Drive never points past the frames on Drive. A failed push is retried at the next checkpoint.
- **Only validated checkpoints reach Drive.** A new `md.cpt` is copied into `state/` only after `gmx dump` has read
  the whole file and its step is past the one already there; otherwise Drive keeps the last good one and the
  rejection is counted in `progress.json`. The checkpoint it replaces is kept as `state/md.prev.cpt`. rclone checks
  each upload's MD5 against Drive, and `status` compares Drive's MD5 of `md.cpt` with the one the node validated.
- **Resume falls back.** If `md.cpt` on Drive does not read in full, the job resumes from `md.prev.cpt` (at most
  10 more minutes lost) and replaces the bad file. If neither reads, it stops with an error rather than restart
  production from zero. CI tests this on every system by truncating `md.cpt` between two nodes.
- **Parts roll every hour** (`-maxh 1 -noappend`), so a part file stops growing after an hour. A resumed run starts a
  new part, and `gmx trjcat` drops the overlapping frames.
- **What an interruption costs.** When Salad reclaims a node, nothing signals `mdrun`. The loss is at most the time
  since the last checkpoint pushed, ≤ 10 min (≈ 5 ns at the ~750 ns/day measured on a 4070 Ti). Salad notices only
  when the job's lease expires and then gives the job to another node, which resumes from Drive; on the first
  night a node that vanished without a log line left its job unclaimed for about 35 min. A container that restarts
  on the same node, or an `mdrun` crash (an intermittent `CUDA error #700` was seen once in production), fails the
  try at once, and the retry usually lands on the same node. Salad counts each as a failed try and gives a job at
  most 4; after that `watch` queues the replica again (below).
- **Completion.** The job writes `state/DONE`, pushes everything (up to 5 attempts), and only then exits 0, so the
  job succeeds only when the replica is complete on Drive. `submit.py` still trusts `state/DONE`, never the job status
  alone: a job that ended without it is queued again.
- **Nothing is ever deleted on Drive.** Every transfer is `rclone copy`, never `sync`.
- **Folders are created once, before any job starts** (`submit` does it in one serial rclone call). Drive allows two
  folders with the same name, and replicas starting together would otherwise race to create them.
- **API load:** a few dozen Drive requests per replica per checkpoint, a few thousand per hour with 27 replicas, far
  below Drive's per-user limit.

## Production length: `control/prod_ns`

Every replica reads `control/prod_ns` before each production segment (at most an hour apart) and resets its own
`md.tpr` with `gmx convert-tpr -nsteps`. So the length can be changed for all replicas at once, including running ones:

```bash
python jobs/submit.py length 85    # every replica now stops at 85 ns
```

A replica already past the new length stops at its next segment and is `DONE`. Keep all replicas at one length:
change it once, after the first speed measurement, and set `endframe = NS × 100` in `analysis/mmpbsa_{gb,pb}.in`.

## One-time setup

1. **Google OAuth client** (once, signed in as moleculardynamicsbio1@gmail.com, in
   [Google Cloud Console](https://console.cloud.google.com/)):
   - Create a project (any name, e.g. `eralpha-antidepressant-md`) and enable the **Google Drive API** for it.
   - *Google Auth Platform*: User type **External**; app name e.g. `eralpha-antidepressant-md rclone`; your own address as support
     and developer contact. Under *Data access* add only the scope `.../auth/drive.file`. Under *Audience* leave the
     status at **Testing** and add moleculardynamicsbio1@gmail.com as the only test user. (Publishing would need a
     public home-page and privacy-policy link, which this project does not have.)
   - *Clients* → *Create client* → type **Desktop app**. Copy the client ID and secret at once: Google shows the
     secret only in the creation dialog (or download its JSON there).
   - **In Testing status the token expires 7 days after you sign in**, and every job's Drive pull then fails. The
     whole run takes about a day, so sign in (step 2) just before `group`. If anything runs past day 7, refresh:
     `rclone config reconnect gdrive:`, then `python jobs/submit.py secrets` (copies the new token into the
     group; Salad restarts its instances, which resume from their checkpoints), then `submit` to requeue any
     replica whose job failed meanwhile.
2. **rclone remote** on your computer (rclone ≥ 1.70), signed in as moleculardynamicsbio1 when the browser opens:
   ```bash
   rclone config create gdrive drive client_id=<ID> client_secret=<SECRET> scope=drive.file
   rclone mkdir gdrive:eralpha-antidepressant-md && rclone lsd gdrive:
   ```
   `group` checks that the remote is type `drive`, scope `drive.file` and uses your own client ID.
3. **Salad.** Put an API key in `.env`, along with the organisation and project names (`SALAD_API_KEY`, `SALAD_ORG`,
   `SALAD_PROJECT`). See `.env.example`.
4. **Image access.** Make the GHCR package public (GitHub → Packages → eralpha-antidepressant-md → Package settings → Change
   visibility). It contains no data and no credentials. If you keep it private, create a classic token with only
   `read:packages` and set `GHCR_USER` and `GHCR_TOKEN` in `.env` before running `group`.

## Before spending: CI validation

`.github/workflows/validate.yml` (on push, or *Actions → validate → Run workflow*) runs the image pinned in
`jobs/submit.py` through the whole pipeline on CPU for every system in `jobs/manifest.yaml`, with short
stages, a SIGKILL during production and a resume on a fresh node. It needs no GPU and no credentials. All 9 jobs
must be green before `group`. If the systems are rebuilt, upload the new `builds.tar.xz` as a new release
(`inputs-v2`, …) and update `INPUTS_TAG` and `INPUTS_SHA256` in the workflow.

## Monitoring

```bash
python jobs/submit.py dashboard     # live page on http://127.0.0.1:8765, refreshed every minute
python jobs/submit.py status        # the same, once, in the terminal
```

Both read three sources: the job queue (job state, tries, job time), Salad (the group and each node: state, CPU, RAM)
and, from Drive, each replica's `state/progress.json`. The node rewrites that file at every equilibration stage and
every checkpoint. It holds the stage, the step and ns secured, the live ns/day, the node and its GPU load and
temperature, and the MD5 of the validated checkpoint. For each replica the page shows:
- progress (equilibration and production);
- speed;
- the node and GPU it runs on, and how many nodes it has used;
- checkpoint age and integrity.

It also flags:
- a running replica silent for more than 25 min;
- any rejected checkpoint, or a mismatch between the checkpoint on Drive and the one the node validated;
- failures;
- a budget that no longer fits the production length;
- GPUs left running with no job.

The dashboard binds to 127.0.0.1 only. The Salad key and Drive token stay in that process, and the page receives
only the summary JSON. Leave it running next to `watch`, in `tmux` or a second terminal.

## Running

```bash
pip install requests pyyaml
python jobs/submit.py manifest                 # jobs/manifest.yaml: 27 replicas with fixed seeds (committed)
python jobs/submit.py inputs --src SYSTEMS_DIR # the built systems/<s>/<p>/build/ (built on Colab)
python jobs/submit.py queue                    # creates the job queue eralpha-antidepressant-md (once)
python jobs/submit.py group                    # creates the group eralpha-antidepressant-md-q on that queue, stopped, 0 replicas
python jobs/submit.py length 90                # writes control/prod_ns
python jobs/submit.py submit                   # queues all 27; creates their folders on Drive first
python jobs/submit.py replicas 1               # one GPU first: the first real job is the test
python jobs/submit.py dashboard                # and open http://127.0.0.1:8765
```

After a rebuild of the image, repin `IMAGE` in `jobs/submit.py` and run `image` to point the group at it.

**First 20 minutes (one GPU, ≈ $0.06).** `status` and Drive should show:
- `state/nodes.txt` with an RTX 4070 Ti (or Ti Super), and `em.done`, `nvt.done`, then the `npt_posres_*` markers
  appearing a few minutes apart.
- the job `running` with no failed tries, and in the Salad portal's container log the worker's `received job` and
  `job_server: job …: run_replica.sh …` lines.

If anything is wrong, `replicas 0` stops all spending. Otherwise:

```bash
python jobs/submit.py watch --max 9            # keeps (pending + running) GPUs, up to 9; stops at the end
```

`watch` checks every 5 minutes. It first queues again every replica whose latest job failed (4 tries used up) or
ended without `state/DONE` on Drive, at most 3 times per replica, and prints an `ALERT` for one that keeps failing.
Then it sets the group's replica count to the number of pending and running jobs (at most `--max`), so no GPU sits
idle while billed, and it stops the group when nothing is left. A running job marks its instance with a higher
deletion cost, so a scale-down removes idle instances first. Leave `watch` running in a terminal (`tmux` or
`nohup`); if it stops, idle GPUs keep billing until you run `replicas 0`, and failed replicas are not requeued.

**After about 1.5 hours** the first replica finishes its first production segment and `status` prints the measured
ns/day, the GPU-hours so far, and the longest production that fits the budget, e.g.

```
measured median 380 ns/day over 1 replica(s): 27 replicas x 90 ns need ≈ 162 GPU-h ≈ $27.50; $30 with a 10% reserve fits 89 ns per replica
```

If the fit is below 90 ns, run `length <fit>` at once: no replica has reached the end of production by then.

`submit` is idempotent: it skips any replica that has `state/DONE` on Drive or a pending or running job. Run it
again (`--only SYSTEM`) for a replica `watch` gave up on, after reading its `data/*.log`: a real failure is a non-zero
exit, for example a grompp error, a crash, or energy minimisation not converging.

**Queue rules to keep in mind.**
- The queue connection is fixed when the group is created. A different queue needs a new group.
- Never delete the queue while jobs are pending or a group uses it: deleting it deletes its jobs and breaks the group.
- `cancel` cancels pending jobs; Salad does not stop a job that is already running. `replicas 0` stops the GPUs;
  each running replica pushes a checkpoint on SIGTERM, and its job fails and is requeued later by `watch`.
- Stop the group only once the queue is empty (`watch` does this): a stopping instance can still take a job.
- If a node's queue worker loses its job stream while `mdrun` keeps going, the job's lease runs out and Salad gives
  the same replica to another node. The first night this happened to 2 of 27 replicas (workers looping on
  "job stream closed" after connection resets). The new node resumes from the newest checkpoint on Drive and
  appends its line to `state/nodes.txt`; before every push, `run_replica.sh` checks that no node's line follows its
  own there, and if one does it stops `mdrun` and exits 3 without pushing, so the two never overwrite each other's
  parts and checkpoint (`tests/drive_resume_test.sh`, takeover case). Images built before this check let both
  nodes push until each finishes; the replica's folder then mixes two trajectories that share everything up to the
  takeover checkpoint. Tell them apart by the host name in each `md.partNNNN.log` and keep one.

## Afterwards

```bash
rclone copy gdrive:eralpha-antidepressant-md/runs runs --transfers 4
```

Then join the parts of each replica before analysis: `gmx trjcat -f md.part*.xtc -o md.xtc` (the same for `.trr`)
and `gmx eneconv -f md.part*.edr -o md.edr`. A part cut short by a crash or interruption runs past the checkpoint
the next part resumes from; both tools keep the later file's frame for a repeated time, which is the right one. Large downloads belong on Colab, which can mount the same Drive directly.
