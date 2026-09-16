# Training the pouring policy

The Mac records and deploys. The in-house NVIDIA GPU trains.

## Host

- `GPU_HOST`: <FILL IN — e.g. user@gpubox, set from the operator's SSH access to the GPU machine>
- `GPU_DIR`: <FILL IN — the remote working directory, as an ABSOLUTE path, e.g.
  `/home/user/lerobot_pour`. Do not use `~/lerobot_pour`: your local shell expands a leading `~`
  in a plain assignment before it ever reaches the remote host, so a `~`-relative `GPU_DIR` would
  silently send your own Mac's home directory path instead of a remote one.>
- Interpreter: <FILL IN — path recorded after running Task 6 step 1's probe. Default (no
  `REMOTE_PY` override) is `$HOME/lerobot-venv/bin/python` on the GPU host; only fill this in if
  you set `REMOTE_PY` to something else, e.g. `/opt/conda/envs/lerobot/bin/python`>
- GPU: <FILL IN — name and memory from `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`>

A `REMOTE_PY` override falls into the same quoting trap as `GPU_DIR`, from the other side: the
interpreter path is sent to the remote shell inside double quotes, and a leading `~` never expands
inside quotes, so `REMOTE_PY=~/lerobot-venv/bin/python` would either be expanded by your own Mac's
shell or arrive as a literal `~` the GPU host refuses to run. Write `$HOME/...` in single quotes,
or an absolute path.

Note: `train_remote.sh` uses `rsync -s` (`--protect-args`) to keep a `GPU_DIR` containing spaces
safe. `--protect-args` requires rsync 3.x on both ends. If the GPU host has an older rsync (check
with `ssh "$GPU_HOST" rsync --version`), either upgrade it or avoid the issue by using a `GPU_DIR`
with no spaces.

## Measured throughput

- Probe: <FILL IN — N> steps in <FILL IN — N> s (run `GPU_HOST=... GPU_DIR=... ./train_remote.sh probe` and record the elapsed seconds it prints)
- Projected 100,000 steps: <FILL IN — probe seconds x 500 / 3600> h

## Stages

Run them in this order. Each stage writes to its own remote directory, because LeRobot
refuses to start a fresh run in a directory that already exists.

| Command | Steps | Remote output | Purpose |
|---|---|---|---|
| `./train_remote.sh probe` | 200 | `outputs/train/pour_v1_probe` | measure throughput; nothing is fetched |
| `./train_remote.sh validate` | 20,000 | `outputs/train/pour_v1` | starts the deployable run |
| `./train_remote.sh full` | 100,000 | `outputs/train/pour_v1` | continues that same run |
| `./train_remote.sh local` | 2,000 | `outputs/train/pour_v1_local` (on the Mac) | Apple GPU fallback, no GPU host |
| `./train_remote.sh fetch` | — | — | copies `pour_v1` back without training |

The probe's checkpoint stays on the GPU host and is never fetched: `pour.py` deploys from
`outputs/train/pour_v1`, so a 200-step checkpoint can never be picked up by mistake.

`full` decides for itself whether to resume: if `outputs/train/pour_v1` already exists on the GPU
host it continues that run from its last checkpoint and raises the step budget to 100,000,
otherwise it starts fresh. Override it with `RESUME=1` (force a resume) or `RESUME=0` (force a
fresh run). If ssh cannot answer the question the stage stops rather than guessing.

Re-running `validate` after it has already produced `pour_v1` will stop with LeRobot's
`FileExistsError`. That is deliberate: move or delete the old `pour_v1` on the GPU host first, or
go straight to `full`, which resumes it.

`./train_remote.sh local` trains on this Mac's GPU (`--policy.device=mps`) with no ssh and no
rsync. It is a way to prove the dataset and the pipeline before booking the GPU host, not a way to
produce a policy worth deploying: it writes to `outputs/train/pour_v1_local`, which `pour.py` does
not read. Override the interpreter it uses with `LOCAL_PY=...` if the venv is not at
`../robot-arm/.venv/bin/python`.

## If the arm stops responding mid-run

The servos hold the last position they were commanded, under torque, when the USB link drops.
They do not go limp, and they do not return to rest. `estop_so101.py` needs that same USB link to
release torque, so it cannot help once the link is gone. Recovery is a power cycle of the arm:
support the arm first, because it will fall as torque dies.
