# Training the pouring policy

The Mac records and deploys. The in-house NVIDIA GPU trains.

## Host

- `GPU_HOST`: <FILL IN — e.g. user@gpubox, set from the operator's SSH access to the GPU machine>
- `GPU_DIR`: <FILL IN — e.g. ~/lerobot_pour, the remote working directory>
- Interpreter: <FILL IN — path recorded after running Task 6 step 1's probe, e.g. ~/lerobot-venv/bin/python>
- GPU: <FILL IN — name and memory from `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`>

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
