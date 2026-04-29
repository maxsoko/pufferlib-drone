# Ubuntu 22.04 GPU Setup for Native Drone Training

This project should train through the native PufferLib v4 `_C` backend. Do not use `--slowly` as a fallback or acceptance path.

## GPU Choice

Default rental target:

- RTX 4090 24GB on RunPod or Vast.ai.
- Ubuntu 22.04.
- CUDA 12.x with `nvcc` available.
- Recent NVIDIA driver.
- 16+ vCPU.
- 64GB RAM.
- 100GB+ persistent disk or volume.

Use Vast.ai if minimizing cost matters most and interruptions are acceptable. Use RunPod if setup friction and uptime are more important. A100 is only needed if 4090 hosts are unreliable or you start larger sweeps. H100 is overkill until vision-heavy or Transformer-scale training is introduced.

## Instance Requirements

Before training, verify:

```bash
nvidia-smi
nvcc --version
python --version
```

Expected:

- `nvidia-smi` sees the GPU.
- `nvcc` is on `PATH`.
- Python is `3.10+`.

## System Packages

Most GPU rental images already include NVIDIA drivers and CUDA. On a clean Ubuntu 22.04 host, install the normal build tools:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  clang \
  libomp-dev \
  ccache \
  git \
  python3-dev \
  python3-pip \
  python3-venv
```

If the host image does not include CUDA/NVCC, switch images rather than hand-installing CUDA unless you specifically want to maintain that machine.

## Python Environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If native build dependencies are missing:

```bash
python -m pip install pybind11 numpy
```

Install the CUDA-enabled PyTorch wheel appropriate for the host image if it is not already present.

## Native GPU Precheck

Run the precheck first:

```bash
bash scripts/validate_native_drone_gpu.sh --precheck-only
```

This should pass on Linux with CUDA/NVCC. It should fail on macOS or CPU-only hosts.

## Native Build and Training Smoke

Run a short native training smoke:

```bash
TIMESTEPS=65536 bash scripts/validate_native_drone_gpu.sh
```

This builds native CUDA `_C` backends for:

- `drone`
- `drone_race`

Then it runs native training without `--slowly`.

## Real Training Order

Train hover/control first:

```bash
bash build.sh drone
python -m pufferlib.pufferl train drone
```

Then train race through the staged bridge curriculum. Use the best hover `.bin` checkpoint from the previous step. The script promotes through one, two, three, then four gates and stops if a stage misses its success/crash thresholds:

```bash
HOVER_WEIGHTS=checkpoints/drone/<run_id>/<checkpoint>.bin \
  bash scripts/train_drone_race_curriculum.sh
```

## Checkpoints

Use a persistent volume. Checkpoints are written under:

```text
checkpoints/<env_name>/<run_id>/*.bin
```

For spot/interruptible hosts, sync checkpoints frequently:

```bash
rsync -av checkpoints/ /persistent/checkpoints/
rsync -av logs/ /persistent/logs/
```

## Edge/Q8 Follow-Up

After a native `.bin` checkpoint exists, run Q8 export and comparison:

```bash
python scripts/export_puffernet_q8.py \
  checkpoints/drone/<run_id>/<checkpoint>.bin \
  checkpoints/drone/<run_id>/<checkpoint>_q8.npz \
  --input-dim 23

WEIGHTS=checkpoints/drone/<run_id>/<checkpoint>.bin \
  bash scripts/eval_drone_hover_edge.sh
```

Race and hover checkpoints both use `--input-dim 23`, so hover weights can warm-start the race curriculum.

## Acceptance Rule

Accepted validation means:

- Native `_C` build succeeds.
- Native training runs without `--slowly`.
- Native eval metrics are recorded.
- Checkpoints are produced and can be used by the edge/Q8 tools.

`--slowly` is debug-only and does not count.