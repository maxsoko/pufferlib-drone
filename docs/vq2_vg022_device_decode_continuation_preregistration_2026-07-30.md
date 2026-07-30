# VQ2 VG022 bitwise device-decode continuation preregistration — 2026-07-30

VG022 is one offline-only continuation from the immutable VG020 epoch-3 state.
It removes only redundant host copies and moves lossless visual-mask expansion
to CUDA. VG021 source batching is rejected and is absent from this variation.

## Frozen identity

- Tag: `vq2_vg022_five_source_device_decode_continuation_001`.
- Parent state SHA-256:
  `3d4694baaeac55184d3672675b4c159ee339adbff45c4b46246a58118ecbb14b`.
- Parent source commit `83a48e63e1982aa636ff2b5136a9ea2451e6ff7d`; completed epoch
  `3`; optimizer updates `31,742`; seed `429094`.
- Preserve model, optimizer, scaler, Python/NumPy/Torch/CUDA RNG, history,
  baseline, best state, configuration, source splits, and every legality/safety
  field exactly.
- Use only the exact runtime embedded in the hash-bound migration state: Python
  `3.12.3`, NumPy `2.1.0`, PyTorch
  `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA `13.1`, RTX 4090, and
  `Linux-6.8.0-90-generic-x86_64-with-glibc2.39`.

## Sole permitted optimization

- Transfer each stored `uint8` 64x64 mask to CUDA before float32 expansion by
  exact `1/255`, and avoid redundant copies of already-contiguous action/valid/
  transition arrays.
- The reconstructed 4,119-value observation must be bitwise identical.
- Execute the five actor calls separately in the original source order. No
  concatenation, vectorized actor batch, padding, stream concurrency, compiled
  graph, loss reordering, or optimizer change is permitted.

## Zero-step parity and throughput gate

Before creating VG022 state or executing epoch 4, the source-locked checker
must use the actual restored next-epoch chunks and verify:

- bitwise observations, actor mean/pre-tanh/next-state, loss, and every
  parameter gradient between legacy and device-decode paths;
- zero optimizer steps and zero state/checkpoint/report writes;
- median complete transition-step wall time (five decodes once plus four
  unchanged forward/backward exposures) improves by at least `1.15x` over the
  legacy host-decode path on the locked Vast runtime.

Any failure rejects VG022 without an unchanged retry.

## Continuation and admission

- Atomically migrate the exact VG020 dynamic payload under VG022 identity, then
  complete epochs `4..12`. Persist and sync state after every epoch.
- Retain exact weights `0.35/0.10/0.10/0.15/0.30`, four agents per source,
  256-step BPTT, four transition exposures, AdamW `1e-5`, weight decay `1e-5`,
  gradient cap `1.0`, and smoothness `1e-4`.
- Final history must be exact epochs `1..12`. Select minimum fixed-weight
  validation and retain VG020's strict overall/VG019 improvement, clean/VG009
  `<=0.02`, later-source `<=0.10`, exact-weight, and `>=4.0x` transition gates.
- Sync parity, epoch states, terminal report/checkpoint/state/log/exit before
  stopping the sole active Vast instance.

VG022 can authorize only a new, separately preregistered teacher-free offline
screen. FlightSim, shadow, Training, and Submission remain unauthorized.
