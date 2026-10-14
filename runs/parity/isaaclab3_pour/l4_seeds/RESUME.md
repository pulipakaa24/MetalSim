# Isaac Franka-Pour seeds 1-2 on the L4 VMs: resume notes

**Finished 2026-10-13.** Seed 2 finished at 18:15 on aditya-isaac and seed 1 at 19:10 on aditya-isaac-2. The
fingerprint matched, and both VMs powered themselves off. The analysis is in docs/PARITY.md §1b.5. The notes
below describe the run and how to repeat it, e.g. for seeds 3-4. Run the launcher from the repo root, or set
`L4_SEEDS_DIR`.

Started 2026-10-13 12:56 PDT. Two detached launchers run on the 16-inch Mac
(`metalsim/parity/isaac_side/isaaclab3/l4_seed_launcher.sh`, snapshot in `scripts/` here). They do not depend on
any Claude session.

| launcher | VM / zone | seeds | check before launch | fallback |
|---|---|---|---|---|
| `launcher_aditya-isaac.pid` | aditya-isaac / us-west1-a (ran seed 0) | 2 | fingerprint recorded only | seed 1 if aditya-isaac-2 never came up |
| `launcher_aditya-isaac-2.pid` | aditya-isaac-2 / us-central1-a (clone of 2026-10-09) | 1 | `fingerprint.sh` must match the seed-0 reference exactly | seed 2 if aditya-isaac never came up |

Each launcher tries `instances start` every 5 min for at most 12 h (until 2026-10-14 00:56). It never creates,
moves, resizes or deletes instances. When the VM is up, it runs `fingerprint.sh`, stages `stage_pour*.sh` and the
asset patch, and launches `stage_pour_seeds.sh` with `POWEROFF=1 POWEROFF_DELAY=1800`. It then polls every 5 min
and copies the train logs here, with the VM user scrubbed to `<vm-user>`. The VM powers itself off 30 min after
its last seed. Seeds 3-4 are not run.

The seed-0 reference that `fingerprint.sh` checks:

* IsaacLab3 at `ae37b028`, clean tree;
* driver 580.178.04;
* Kit's omni client 2.72.3-release.7151;
* rsl_rl 5.4.1;
* stock `max_triangle_pairs: int = 1_000_000`, not overridden by franka_pour.

## Check status

    cat runs/parity/isaaclab3_pour/l4_seeds/status_*.txt

The first word of each line is the state. RETRYING, BOOTING, STARTED and TRAINING are in progress (TRAINING shows
progress and the expected finish). DONE means the logs are here. FAILED gives the reason: auth expired (run
`gcloud auth login`, then restart the launcher), VM never started, fingerprint mismatch, or launch failed.
Full history: `launcher_<vm>.log`. Liveness: `ps -p $(cat launcher_aditya-isaac.pid) $(cat launcher_aditya-isaac-2.pid)`.

To restart one launcher, run it detached (it is safe while its VM is running: the start is a no-op, and it
re-stages the scripts and launches):

    python3 -c 'import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' /bin/bash \
      runs/parity/isaaclab3_pour/l4_seeds/scripts/l4_seed_launcher.sh aditya-isaac-2 us-central1-a 1 1 2 aditya-isaac &

To stop everything: `kill` both PIDs. A running seed still powers its VM off when it ends.

## Fetch results

The launchers copy `train_1024_curve_s<seed>.log` here. To fetch by hand while a VM is up:

    gcloud compute scp --zone <zone> --tunnel-through-iap --project mcmc-193822 \
      <vm>:parity3/pour/train_1024_curve_s1.log .

Then scrub the logs with `sed -i '' 's#/home/[^/]*/#/home/<vm-user>/#g'`.

## Analysis and PR

    python3 -m metalsim.learn.franka_pour_curve seeds \
      --mac runs/franka_pour/logs/m414_1024_log.jsonl runs/franka_pour/seeds/s{1,2,3,4}/log.jsonl \
      --isaac runs/parity/isaaclab3_pour/train_1024_curve.log runs/parity/isaaclab3_pour/l4_seeds/train_1024_curve_s{1,2}.log \
      --json runs/franka_pour/seeds/summary_with_isaac3.json

1. Move the two logs next to `train_1024_curve.log`.
2. Update docs/PARITY.md §1b.5 (ours n=5 vs Isaac n=3 per window, with a verdict, and the versions from
   `fingerprint_<vm>.txt`).
3. Open a PR to main from a branch off main (`gh pr create --repo pulipakaa24/MetalSim --base main`).
