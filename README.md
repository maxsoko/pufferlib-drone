# PufferLib Drone Race

This branch is the native PufferLib v4 drone race workspace. The active goal is to build an autonomous drone policy that completes gates in strict order under the official virtual qualifier constraints.

`PRD.md` is the source of truth for product direction, scope, promotion gates, and current status.

## Program Goal

Win, or place competitively enough to credibly contend for winning, by submitting a fully autonomous TS-002-compliant drone racing stack that completes the official DCL course in strict gate order with a competitive valid time.

This is not considered complete when a native checkpoint scores well. Completion requires end-to-end SITL/MAVLink + camera validation, no privileged native state, no human input during the run, command-rate/heartbeat compliance, and deterministic run reports.

## Current Direction

- Native PufferLib v4 `_C` is the required training and evaluation path.
- `--slowly` / PyTorch backend runs are debug-only and do not count as acceptance.
- `ocean/drone` is the hover/control-prior environment.
- `ocean/drone_race` is the race environment, now using representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate actions.
- `drone_race` has two training interfaces: legacy native motor/privileged observations by default, and `interface_mode = 1` for TS-002-shaped telemetry/vision observations plus normalized body velocity/yaw-rate policy actions.
- Race timing targets the qualifier spec: `120 Hz` physics and `480s` maximum run duration.
- Final competition integration should go through MAVLink v2 over UDP via a MAVSDK-compatible SITL boundary.
- Edge inference work starts with quantized mixed-precision PufferNet, not Transformers.

## Key Files

- `PRD.md`: active program requirements and status.
- `docs/gpu_setup_ubuntu_22_04.md`: Linux GPU setup and native training runbook.
- `config/drone.ini`: native hover/control-prior config.
- `config/drone_race.ini`: native race config.
- `config/drone_race_competition.ini`: competition-shaped training profile over the compiled `drone_race` backend.
- `config/sitl_competition_acceptance.json`: frozen SITL acceptance/promotion thresholds and robust gate-pass defaults.
- `config/detector_stress_thresholds.json`: deterministic detector stress-suite thresholds.
- `ocean/drone/`: representative hover drone dynamics and native binding.
- `ocean/drone_race/`: native race environment, gate sequencing, metrics, and representative race physics.
- `scripts/eval_drone_hover_native.py`: native hover eval smoke.
- `scripts/eval_drone_race_native.py`: native race eval smoke.
- `scripts/validate_native_drone_gpu.sh`: Linux CUDA/NCCL native validation runner.
- `scripts/drone_sitl_adapter.py`: MAVLink/SITL scaffold.
- `scripts/drone_policy_contract.py`: shared 23-float observation and 4-action velocity/yaw-rate contract for native training and SITL setpoint decoding.
- `scripts/drone_camera_receiver.py`: TS-002 JPEG-over-UDP camera packet parser/reassembler.
- `scripts/drone_gate_detector.py`: first-pass square-gate detector over TS-002 camera frames.
- `scripts/drone_visual_servo.py`: TS-002 gate-corner pose and visual-servo command helper.
- `scripts/drone_sitl_competition_smoke.py`: integrated SITL smoke loop that connects telemetry, camera, gate detection, visual-servo/policy control, and JSON/CSV report output.
- `scripts/sitl_stream_probe.py`: fast UDP preflight probe for MAVLink (`14540`) and camera (`5600`) traffic presence.
- `scripts/mock_ts002_stream.py`: local TS-002 mock telemetry/camera stream generator for offline integration tests.
- `scripts/policy_callable_gate_pid.py`: deterministic competition-shaped policy callable for `--control-mode policy`.
- `scripts/policy_callable_checkpoint.py`: checkpoint-backed policy callable for `--policy-callable`.
- `scripts/sitl_udp_capture.py`: captures MAVLink/camera UDP packets into deterministic JSONL replay events.
- `scripts/sitl_udp_replay.py`: replays captured JSONL MAVLink/camera events with deterministic impairment injection.
- `scripts/sitl_replay_regression.py`: deterministic replay runner with baseline-vs-policy comparison and locked acceptance.
- `scripts/run_official_gate1_validation.py`: one-command official-traffic validator (probe first, then smoke if traffic exists) with deterministic summary artifact.
- `scripts/watch_official_gate1_validation.py`: poll/wait runner that keeps probing for official traffic and auto-triggers gate-1 validation when traffic appears.
- `scripts/detector_stress_suite.py`: synthetic perception stress runner (noise/blur/compression/occlusion/scale).
- `scripts/native_transfer_readiness_report.py`: aggregates native eval + SITL replay comparison into transfer-readiness report.
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
python scripts/eval_drone_race_native.py --env-name drone_race_competition --steps 256
```

Race physics regression:

```bash
bash scripts/test_drone_race_native_regressions.sh
```

Q8 edge benchmark:

```bash
bash scripts/benchmark_puffernet_edge.sh
```

Checkpoint eval report:

```bash
python scripts/eval_drone_race_checkpoint.py \
  checkpoints/drone_race/<run_id>/<checkpoint>.bin \
  --label r3_candidate \
  --json-path logs/drone_race_r3_candidate_eval.json \
  --csv-path logs/drone_race_r3_candidate_eval.csv \
  --env.num-gates 3 \
  --env.gate-radius 1.9 \
  --env.gate-spacing 4.5 \
  --env.gate-lateral-amplitude 1.0 \
  --env.start-offset 1.75 \
  --env.crash-height -1.0 \
  --env.strict-missed-gate 0
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

Then train race through the bridge curriculum. Pass the latest hover checkpoint if you have one. The runner now promotes through one, two, three, then four gates and stops if a stage misses its success/crash thresholds:

```bash
HOVER_WEIGHTS=checkpoints/drone/<run_id>/<checkpoint>.bin \
  bash scripts/train_drone_race_curriculum.sh
```

For competition-shaped policy training, build the same compiled backend and train the profile:

```bash
bash build.sh drone_race
python -m pufferlib.pufferl train drone_race_competition
```

This profile uses `backend_env_name = drone_race`, so checkpoints/logs are separated under `drone_race_competition` while the native backend remains the compiled `drone_race` environment.

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
  --input-dim 23
```

## Current Status

- Native hover CPU build and eval smoke pass locally.
- Native race CPU build, eval smoke, and gate-crossing/physics/timing regressions pass locally.
- Race physics now use quadrotor motor/RK4 dynamics.
- Competition-shaped `drone_race` interface mode and Python policy contract pass local regression/unit coverage.
- Q8 benchmark and hover closed-loop comparison harness compile locally.
- MAVLink/SITL dry-run scaffold, telemetry parser, TS-002 camera receiver, and visual-servo geometry helpers pass local tests.
- Integrated SITL competition smoke runner and square-gate detector pass local unit coverage.
- Gate-pass logic now uses confidence gating, hysteresis/rearm bands, debounce over missed detections, cooldown, and deterministic rejection counters in artifacts.
- On 2026-05-22, `scripts/drone_sitl_competition_smoke.py` produced deterministic JSON/CSV artifacts locally, but with zero inbound telemetry and zero camera frames (`ordered_gate_passes=0`), so first-gate proof is still blocked on live simulator traffic.
- The smoke runner now loads frozen acceptance thresholds from `config/sitl_competition_acceptance.json` and supports CLI overrides for gate-pass and packet-health checks.
- A fast preflight probe now exists to verify inbound MAVLink/camera traffic before smoke runs.
- On 2026-05-30, strict acceptance passed end-to-end against a local TS-002 mock stream (`scripts/mock_ts002_stream.py`), with `ordered_gate_passes=1` and nonzero telemetry/camera traffic in deterministic artifacts under `logs/sitl/competition_smoke_gate1_mocked_required.{json,csv}`.
- Policy-mode wiring is now usable without official traffic through `--policy-callable scripts/policy_callable_gate_pid.py:infer`.
- Deterministic UDP capture now writes sidecar metadata with `capture_id` and SHA256, and replay can enforce metadata hash checks.
- Replay-driven policy smoke acceptance is now reproducible from captured UDP events (`logs/sitl/capture_mock_24s.jsonl` -> `logs/sitl/competition_smoke_policy_replay_required.json`) using calibrated gate-pass thresholds for that clip.
- Deterministic replay regression runner now supports repeated runs (`N`) and degraded replay profile injection (`loss/reorder/latency/jitter`) for locked promotion metrics.
- Detector stress suite now provides deterministic pass/fail metrics for noise, blur, compression, occlusion, and scale.
- On 2026-05-30, Linux GPU native validation succeeded on Vast RTX 4090 (`scripts/validate_native_drone_gpu.sh` precheck + smoke complete), and short `drone_race_competition` training produced checkpoints under `checkpoints/drone_race_competition/1780187064493/`.
- On 2026-05-30, checkpoint eval artifacts were generated in `logs/drone_race_competition_eval.json` and `logs/drone_race_competition_eval.csv`.
- Gate-pass defaults are now frozen in `config/sitl_competition_acceptance.json` at calibrated replay values (`pass_range_m=2.5`, `rearm_range_m=2.8`, `arm_range_m=3.2`, `confidence_pass_min=0.4`, `min_consecutive_pass_frames=1`, `max_missed_detection_frames=5`).
- Canonical smoke artifacts now pass from replayed telemetry+camera traffic without CLI gate-pass overrides:
  - `logs/sitl/competition_smoke_gate1.json`
  - `logs/sitl/competition_smoke_gate1.csv`
- Deterministic replay regression now passes at `N=30` for both baseline and checkpoint policy:
  - Normal profile: `baseline_valid_rate=1.0`, `policy_valid_rate=1.0` (`logs/sitl/replay_regression_summary_policy_4090_tuned_normal.json`)
  - Degraded Profile A (`2%` loss, `5%` reorder, `30ms` latency, `10ms` jitter): `baseline_valid_rate=1.0`, `policy_valid_rate=1.0` (`logs/sitl/replay_regression_summary_policy_4090_tuned_degraded_a.json`)

## Next Required Work

1. Validate strict acceptance against official simulator telemetry/camera traffic (not just local mock/replay) and reproduce first-gate proof there.
2. Keep frozen acceptance config stable and rerun `N=30` normal/degraded replay after perception/controller changes.
3. Retrain/extend `drone_race_competition` on Linux GPU and verify policy remains non-inferior on the same replay set.
4. Run detector stress suite before changing detector parameters, and keep threshold config frozen across commits.
5. Use metadata-verified capture/replay artifacts for regression and promotion evidence.
6. Continue R3 native stabilization and Q8 work as support tracks after official SITL validation is in place.

Official-traffic gate-1 validation (single command):

```bash
python scripts/run_official_gate1_validation.py \
  --summary-json-path logs/sitl/official_gate1_validation_summary.json \
  --probe-json-path logs/sitl/stream_probe_official.json \
  --smoke-json-path logs/sitl/competition_smoke_gate1_official.json \
  --smoke-csv-path logs/sitl/competition_smoke_gate1_official.csv
```

Official-traffic watch mode (hands-off until traffic appears):

```bash
python scripts/watch_official_gate1_validation.py \
  --poll-interval-s 15 \
  --max-wait-s 1800 \
  --watch-summary-json-path logs/sitl/official_gate1_watch_summary.json \
  --summary-json-path logs/sitl/official_gate1_validation_summary.json \
  --probe-json-path logs/sitl/stream_probe_official.json \
  --smoke-json-path logs/sitl/competition_smoke_gate1_official.json \
  --smoke-csv-path logs/sitl/competition_smoke_gate1_official.csv
```

## Offline Regression Workflow

1. Start local mock stream:

```bash
python scripts/mock_ts002_stream.py --duration 40 \
  --json-path logs/sitl/mock_stream_run.json
```

2. Capture deterministic UDP events:

```bash
python scripts/sitl_udp_capture.py \
  --duration 10 \
  --output-path logs/sitl/capture_mock_10s.jsonl \
  --metadata-path logs/sitl/capture_mock_10s.meta.json \
  --generator-params-json '{"source":"mock_ts002_stream","camera_fps":30}' \
  --require-mavlink --require-camera \
  --json-path logs/sitl/capture_mock_10s.json
```

3. Replay captured traffic for deterministic regression:

```bash
python scripts/sitl_udp_replay.py \
  --input-path logs/sitl/capture_mock_10s.jsonl \
  --metadata-path logs/sitl/capture_mock_10s.meta.json \
  --require-hash-match \
  --loops 2 \
  --speed 1.0 \
  --json-path logs/sitl/replay_mock_10s.json
```

4. Run policy-mode smoke:

```bash
python scripts/drone_sitl_competition_smoke.py \
  --endpoint udpin:0.0.0.0:14540 \
  --control-mode policy \
  --policy-callable scripts/policy_callable_gate_pid.py:infer \
  --duration 12 \
  --acceptance-config config/sitl_competition_acceptance.json \
  --require-telemetry --require-camera --min-gate-passes 1 \
  --gate-pass-range-m 1.35 \
  --json-path logs/sitl/competition_smoke_policy_mocked_required.json \
  --csv-path logs/sitl/competition_smoke_policy_mocked_required.csv
```

5. Run repeated replay regression and baseline-vs-policy comparison:

```bash
python scripts/sitl_replay_regression.py \
  --input-path logs/sitl/capture_mock_10s.jsonl \
  --runs 30 \
  --acceptance-config config/sitl_competition_acceptance.json \
  --run-policy \
  --policy-callable scripts/policy_callable_gate_pid.py:infer \
  --loss-rate 0.02 --reorder-rate 0.05 --latency-ms 30 --jitter-ms 10 \
  --summary-json logs/sitl/replay_regression_summary.json
```
