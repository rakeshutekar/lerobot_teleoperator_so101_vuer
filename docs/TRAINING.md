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

Note: `train_remote.sh` uses `rsync -s` (`--protect-args`) to keep a `GPU_DIR` containing spaces
safe. `--protect-args` requires rsync 3.x on both ends. If the GPU host has an older rsync (check
with `ssh "$GPU_HOST" rsync --version`), either upgrade it or avoid the issue by using a `GPU_DIR`
with no spaces.

## Measured throughput

- Probe: <FILL IN — N> steps in <FILL IN — N> s (run `GPU_HOST=... GPU_DIR=... ./train_remote.sh probe` and record the elapsed seconds it prints)
- Projected 100,000 steps: <FILL IN — probe seconds x 500 / 3600> h

## Stages

| Command | Steps | Purpose |
|---|---|---|
| `./train_remote.sh probe` | 200 | measure, do not deploy |
| `./train_remote.sh validate` | 20,000 | first deployable checkpoint |
| `./train_remote.sh full` | 100,000 | the policy worth using |

`RESUME=1 ./train_remote.sh full` continues from the last checkpoint after an
interruption. `./train_remote.sh fetch` copies checkpoints back without training.
