# VQ2 VG021 accelerated five-source continuation preregistration — 2026-07-30

VG021 is one offline-only continuation of VG020 from its immutable atomic
epoch-3 boundary. It exists solely to remove measured host decoding and
underfilled-GPU overhead. It does not alter the actor ABI, datasets, source
weights, recurrent horizon, optimizer, RNG streams, loss, admission rule, or
runtime safety boundary.

## Frozen identity

- Tag: `vq2_vg021_five_source_accelerated_continuation_001`.
- Parent state SHA-256:
  `3d4694baaeac55184d3672675b4c159ee339adbff45c4b46246a58118ecbb14b`.
- Parent source commit: `83a48e63e1982aa636ff2b5136a9ea2451e6ff7d`.
- Parent state must report tag
  `vq2_vg020_variable_gate_five_source_refit_001`, schema
  `vq2_variable_gate_five_source_refit_state_v1`, status `training`, completed
  epoch `3`, optimizer updates `31,742`, best epoch `3`, and the exact frozen
  runtime/config/safety contracts.
- Seed remains `429094`. The Python, NumPy, Torch CPU, and all CUDA RNG states
  are restored from the parent; there is no reseeding after migration.
- Runtime remains the original VG020 platform: Python `3.12.3`, NumPy `2.1`,
  PyTorch `2.10.0a0+nv26.01`, CUDA `13.1`, RTX 4090, and
  `Linux-6.8.0-90-generic-x86_64-with-glibc2.39`.

## Permitted throughput changes

1. Transfer the stored `uint8` visual mask to CUDA before lossless float32
   expansion by `1/255`. The reconstructed actor observation must be bitwise
   identical to the legacy host-decode result.
2. Concatenate source chunks only when their recurrent time horizons are
   identical, execute one actor call for that group, then split outputs and
   next states back by source. Chunks with different horizons remain in
   separate calls; no padding or extra recurrent step is permitted.
3. All five source losses are still computed separately and combined at exact
   normalized weights `0.35/0.10/0.10/0.15/0.30` on every optimizer update.

No other training change is permitted.

## Pre-update parity and throughput gate

Before the first epoch-4 optimizer step, run a command-free checker on the
actual restored next-epoch chunks. It must verify:

- bitwise decoded observations;
- maximum mean/pre-tanh/next-state error `<=1e-6`;
- loss error `<=1e-7` and maximum parameter-gradient error `<=1e-5` under the
  locked CUDA autocast path;
- no model, optimizer, scaler, RNG, state, dataset, report, or checkpoint write;
- optimized forward/backward median wall time at least `1.5x` faster than the
  legacy path on the checked production-shape batch.

Any failure rejects VG021 before an optimizer step. Do not retry unchanged.

## Continuation and admission

- Atomically create the VG021 state by replacing only source/tag/runtime
  identity around the preserved dynamic VG020 payload. Verify the migrated
  model, optimizer, scaler, RNG, history, baseline, best state, and update count
  before starting epoch 4.
- Complete epochs `4..12`. The final history must contain each epoch `1..12`
  exactly once and preserve the first three epoch records byte-for-byte as
  loaded Python values.
- Retain four agents per source, 256-step BPTT, four transition exposures,
  AdamW learning rate/weight decay `1e-5/1e-5`, gradient cap `1.0`, and
  smoothness `1e-4`.
- Select the minimum fixed-weight five-source validation across all 12 epochs.
  Numerical admission retains VG020's rules: strict overall and VG019
  improvement from the frozen baseline; clean and VG009 weighted MSE `<=0.02`;
  VG012, VG016, and VG019 each `<=0.10`; exact source weights; and minimum
  transition exposure `>=4.0x`.
- Save resumable state after every epoch and continuously sync it plus the
  runner log to persistent local evidence storage. Sync the terminal report,
  checkpoint, state, parity report, log, and exit marker before stopping the
  sole active Vast instance.

VG021 is offline numerical-fit evidence only. Even if admitted, it authorizes
only one separately preregistered fresh teacher-free deterministic screen. It
does not authorize FlightSim, shadow, VQ2 Training, or Submission.
