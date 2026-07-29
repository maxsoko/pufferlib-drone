# VQ2 VG004 variable-gate recurrent BC preregistration — 2026-07-29

## Decision and prerequisites

Run one offline CUDA fit tagged `vq2_vg004_variable_gate_recurrent_bc_001` on
the admitted VG003 corpus. The immutable dataset report and metadata SHA-256
values are `b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85`
and `7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64`.
The corpus contains 2,396,907 causal records, 256/256 oracle finishes, zero
collision, exact action-history alignment, and no stored privileged or total-
count value. This training run does not itself admit a deployment checkpoint;
that requires the separately preregistered teacher-free closed-loop screen.

## Fixed model and split

- Seed `429031`; PyTorch deterministic cuDNN mode and float32 matmul precision
  `high`, with CUDA float16 autocast/gradient scaling.
- Actor class `VQ2PhaseRecurrentActor`: the frozen 64x64 mask CNN and legal
  sensor encoder, one learned phase embedding, exactly one 256-wide GRU, one
  joint four-output tanh Gaussian action head. Deterministic deployment uses
  the mean. There is no critic, decoder, course count, planner, selected
  contour, pose, teacher, blend, action override, or fallback in the model.
- Input width 4,119: mask 4,096 + legal tail 22 + one causal 4 Hz held scalar
  `clamp(active_gate_index, 0, 16) / 16`.
- Agents 0--223 train; agents 224--255 validate. The construction-time exact-
  uniform assignment gives 28 training and four validation episodes at each
  count 5 through 12. Selection uses only lowest finite validation weighted
  action MSE; no closed-loop or sealed-test result selects an epoch.

## Fixed optimization

- 12 epochs; agent batch 8; learning rate `3e-4`; AdamW weight decay `1e-5`;
  gradient norm cap 1.0; temporal smoothness coefficient `1e-4`; normalized
  action-channel weights `(1, 1, 4, 1)`.
- Time-major BPTT windows are exactly 256 steps. Each episode starts from a
  zero GRU state; recurrent state is carried causally across every consecutive
  window through the episode's complete valid prefix and detached only at the
  256-step optimization boundary.
- A transition row is a causal increase in the stored held public phase,
  including an increase on the first row after a window boundary. Every batch
  window containing at least one such increase is optimized three times from
  the same incoming recurrent state. Thus every agent-window containing an
  official index increment receives exactly 3x exposure; all other batch
  windows receive one update. Each epoch report must prove a minimum
  transition-agent-window exposure ratio of at least 3.0.
- Validation replays each complete held-out prefix sequentially in 256-step
  chunks with carried recurrent state. The report includes overall and exact
  transition-row error, optimizer updates, window/exposure counts, and the
  selected epoch.

## Resume and outputs

Output is
`logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg004_variable_gate_recurrent_bc_001`.
Before the first update the trainer atomically writes `training_state.pt` with
the exact Git commit, source and dataset hashes, runtime manifest, full config,
model/optimizer/scaler states, and Python/NumPy/Torch/CUDA RNG states. It
atomically replaces that state only after a complete epoch. An interruption
may resume only an exact state with status `training`; any incomplete epoch is
discarded by loading the last completed snapshot. Once state exists, the Vast
wrapper must not install, test, or alter runtime dependencies.

The final `policy_best.pt` and `report.json` are written once training
completes. The checkpoint includes the source/runtime contract, full history,
split, minimum transition exposure, safety fields, and no optimizer-only or
privileged deployment component. The epoch state is then sealed with status
`completed` plus checkpoint/report hashes.

## Frozen sources before commit/runtime locking

- `scripts/train_vq2_variable_gate_recurrent_bc.py` — `c0ab4fccbdff93ca27476a2d09b34f72d70071afa039f8701cdf9f4b37227cba`
- `scripts/run_vq2_vg004_vast.sh` — `9953ce42423ecf42e7c28257c0e0476f547c5fab85c8f26450a8dec7d6cb24c4`
- `tests/test_train_vq2_variable_gate_recurrent_bc.py` — `c31dba1daaea5d693c5245042f322f6a1c008687855d1698166ce5cf87f4656d`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_recurrent.py` — `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `scripts/train_vq2_recurrent_bc.py` — `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`

The trainer hashes itself, this preregistration, the runner, actor modules,
shared loss utilities, and VG003 report/metadata into epoch-zero state. The
new Git commit and Vast runtime are recorded before the first update.

## Safety and next gate

This is legal-observation native offline work. Teacher blend, FlightSim
packets, and sealed N712 accesses remain zero; Submission is forbidden. After
VG004 completes, preregister one teacher-free deterministic screen of 256 new
randomized courses, 64 at each count 5/8/11/12. Stage-2 acceptance is at least
90% full-course completion overall with zero crash. If it fails at a localized
transition, preserve the checkpoint and move to source-balanced full-course
DAgger rather than silently changing or rerunning VG004.
