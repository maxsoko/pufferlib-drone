# PufferLib Drone Race

This branch is the native PufferLib v4 drone race workspace. The active goal is to build an autonomous drone policy that completes gates in strict order under the official virtual qualifier constraints.

`PRD.md` is the source of truth for product direction, scope, promotion gates, and current status.

## Current Direction

- Native PufferLib v4 `_C` is the required training and evaluation path.
- `--slowly` / PyTorch backend runs are debug-only and do not count as acceptance.
- `ocean/drone` is the hover/control-prior environment.
- `ocean/drone_race` is the race environment, now using representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate actions.
- Race timing targets the qualifier spec: `120 Hz` physics and `480s` maximum run duration.
- Final competition integration should go through MAVLink v2 over UDP via a MAVSDK-compatible SITL boundary.
- Edge inference work starts with quantized mixed-precision PufferNet, not Transformers.

## Key Files

- `PRD.md`: active program requirements and status.
- `docs/gpu_setup_ubuntu_22_04.md`: Linux GPU setup and native training runbook.
- `config/drone.ini`: native hover/control-prior config.
- `config/drone_race.ini`: native race config.
- `ocean/drone/`: representative hover drone dynamics and native binding.
- `ocean/drone_race/`: native race environment, gate sequencing, metrics, and representative race physics.
- `scripts/eval_drone_hover_native.py`: native hover eval smoke.
- `scripts/eval_drone_race_native.py`: native race eval smoke.
- `scripts/validate_native_drone_gpu.sh`: Linux CUDA/NCCL native validation runner.
- `scripts/drone_sitl_adapter.py`: MAVLink/SITL scaffold.
- `src/puffernet_q8.h`: Q8 mixed-precision PufferNet runtime scaffold.

## Local CPU Checks

Local macOS CPU builds are only correctness smoke tests. They do not replace Linux CUDA training.

```bash
SDKROOT="$(xcrun --show-sdk-path)" \
CC="/opt/homebrew/opt/llvm/bin/clang" \
CXX="/opt/homebrew/opt/llvm/bin/clang++" \
CPATH="/opt/homebrew/opt/libomp/include:${CPATH:-}" \
LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${LIBRARY_PATH:-}" \
bash build.sh drone_race --cpu

python scripts/eval_drone_race_native.py --steps 256
```

Race physics regression:

```bash
bash scripts/test_drone_race_native_regressions.sh
```

Q8 edge benchmark:

```bash
bash scripts/benchmark_puffernet_edge.sh
```

## Linux GPU Training

Use Ubuntu 22.04 with CUDA/NVCC. A single RTX 4090 24GB is the default practical target.

Precheck:

```bash
bash scripts/validate_native_drone_gpu.sh --precheck-only
```

Short native training smoke:

```bash
TIMESTEPS=65536 bash scripts/validate_native_drone_gpu.sh
```

Train hover first:

```bash
bash build.sh drone
python -m pufferlib.pufferl train drone
```

Then train race:

```bash
bash build.sh drone_race
python -m pufferlib.pufferl train drone_race
```

## Edge Inference

The active edge target is the existing PufferNet architecture:

```text
DefaultEncoder -> MinGRU -> DefaultDecoder
```

Current Q8 path:

- `int8` weights and dynamic activations.
- Per-output-channel scales.
- `int32` accumulators.
- Mixed-precision recurrent state/gates for the first parity milestone.
- Closed-loop validation required before promotion.

After a native `.bin` checkpoint exists, export Q8 weights:

```bash
python scripts/export_puffernet_q8.py \
  checkpoints/drone_race/<run_id>/<checkpoint>.bin \
  checkpoints/drone_race/<run_id>/<checkpoint>_q8.npz \
  --input-dim 25
```

## Current Status

- Native hover CPU build and eval smoke pass locally.
- Native race CPU build, eval smoke, and gate-crossing/physics/timing regressions pass locally.
- Race physics now use quadrotor motor/RK4 dynamics.
- Q8 benchmark and hover closed-loop comparison harness compile locally.
- MAVLink/SITL dry-run scaffold passes locally.
- Native CUDA/NCCL validation still needs to run on Linux GPU hardware.

## Next Required Work

1. Run `scripts/validate_native_drone_gpu.sh` on a Linux CUDA machine.
2. Train native `drone` hover/control prior.
3. Train native `drone_race` only after hover/control metrics are stable.
4. Add telemetry parsing and model-output contracts to the SITL adapter.
5. Add fake quantization during native RL fine-tuning.
