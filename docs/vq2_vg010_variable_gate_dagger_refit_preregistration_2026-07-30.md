# VQ2 VG010 source-balanced recurrent DAgger refit preregistration — 2026-07-30

## Authority and bounded decision

VG009 is admitted as a legal failure-state label corpus by independent local
evidence SHA-256
`71c4898f7a7a306b0f6def06bda63d5192de456c8a849e1f08f83ba908cd3091`.
It contains 123,992 source-locked VG005-visited records from 512 terminal
episodes, reaches Gate 1 in 483/512 and Gate 2 in 32/512, and passes every hard
crash/count/transport/layout/phase predicate. That admission authorizes one
new offline refit, not a policy screen or any simulator action.

Run exactly one fit tagged
`vq2_vg010_variable_gate_source_balanced_refit_001` with seed `429050`. Start
from the immutable VG005 checkpoint; never read VG007/VG008 arrays and never
train from a random actor. VG010 may emit one candidate checkpoint and a
numerical admission decision only. A non-admitted result is final under this
tag and cannot be retried unchanged.

## Immutable data and actor boundary

- Parent VG005 checkpoint SHA-256:
  `f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3`;
  report SHA-256:
  `bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b`.
- Clean VG003 report SHA-256:
  `b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85`;
  metadata SHA-256:
  `7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64`.
  Agents 0--223 train and 224--255 validate.
- Failure-state VG009 report SHA-256:
  `72c1e41d128f16ff082362e0dac2fb0cea36eef1a7d65e8e96f8ec31c3f50637`;
  metadata SHA-256:
  `da20bd4ef93b7371712880787226a7f41b23785841d7307d501e0b7a8f829468`.
  Agents 0--447 train and 448--511 validate.

Both sources reconstruct the same 4,119-value public actor input: 4,096
quantized mask values, the frozen 22-value legal camera/IMU/action/timing tail,
and one causal 4 Hz held official progress scalar. The candidate remains the
single recurrent full-output `VQ2PhaseRecurrentActor`; its four deterministic
outputs are never blended, clipped, scheduled, or replaced by a teacher.
Native state exists only in the already-materialized offline oracle labels.

## Source-balanced optimization contract

- Twelve epochs; AdamW learning rate `3e-5`, weight decay `1e-5`, gradient cap
  `1.0`, smoothness weight `1e-4`, and action weights `(1, 1, 4, 1)`.
- Every optimizer update pairs one clean chunk with one VG009 chunk and assigns
  exactly `0.5` normalized objective weight to each source. Loss is normalized
  within each source before mixing, so VG003's larger raw record volume cannot
  dilute VG009. The audit must report equal cumulative source weight on every
  epoch.
- Source batch size is four agents. Every clean training episode is traversed
  once per epoch. VG009 is independently shuffled and cycles only at complete
  agent-batch boundaries as needed to supply every paired clean update.
- Recurrent state begins at the genuine episode boundary, is carried through
  consecutive time-major 256-step BPTT chunks, and is detached only between
  chunks. A chunk containing a valid public phase increment receives exactly
  three updates from the same incoming recurrent state; all other chunks
  receive one.
- CUDA float16 autocast and gradient scaling are used with deterministic
  cuDNN settings. Epoch-zero state freezes the Git commit, sources, evidence,
  runtime, model/optimizer/scaler, all RNG states, splits, config, and safety
  contract before the first update.

The unchanged VG005 parent is epoch zero. The selected candidate is the lowest
finite mean of clean and VG009 validation weighted MSE, with equal `0.5/0.5`
source weight. Numerical admission requires a child epoch, improvement over
the parent on that balanced score, strict improvement on VG009 validation,
clean validation weighted MSE no greater than `0.02`, exact equal-source
weight audit, and at least three transition-window exposures. These are only
fit-integrity gates; they do not establish flight performance.

## Resume, preflight, and next gate

Output is
`logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg010_variable_gate_source_balanced_refit_001`.
Only an exact identity-matched state with status `training` may resume from its
last completed epoch. Once state exists, the runner performs no install,
build, test, or runtime mutation. A completed resume also verifies current
source commit/hashes plus the bound report, checkpoint, and completed state.

Before epoch zero, the wrapper requires the exact dataset/parent/admission
hashes, CUDA, a supported GPU, at least 32 visible CPUs, approximately 64 GiB
RAM, 20 GiB free disk, `clang`, OpenMP headers, and `ccache`; it then passes
both native regression suites, a fresh float32 vision-native build/ABI check,
and focused recurrent/refit tests. The CPU-only test reductions use 16 OpenMP
and MKL threads on the replacement AMD host; actual CUDA training remains at
the runtime default.

If and only if VG010 is numerically admitted, preregister a fresh teacher-free
deterministic policy screen before executing it. The screen must use new
randomized courses and preserve the legal full-output recurrent boundary.
VG010 itself authorizes no screen execution, Windows shadow, Training run, or
Submission.

## Frozen source surface

- `scripts/train_vq2_variable_gate_dagger_refit.py` —
  `afc7ea57da09365c3f6f98ef3f2faaf520dc16c685548645be96f743a08d00fb`
- `scripts/run_vq2_vg010_vast.sh` —
  `76178b043365e6e4cddf17f6a471c44d5df2eb6e3641efe11095624bd765386c`
- `tests/test_train_vq2_variable_gate_dagger_refit.py` —
  `c8d0b306e3f36c23660391c6e59109c51b94d6f3d35c44db1f7c1d80ab2ac841`
- `scripts/train_vq2_variable_gate_recurrent_bc.py` —
  `940b9cea940b220a3cc112862b6243fc92e40d87588618464d484d48506b0618`
- `scripts/train_vq2_recurrent_bc.py` —
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`
- `pufferlib/vq2_informed.py` —
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_recurrent.py` —
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` —
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- persistent goal prompt —
  `052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`
- VG009 admission evidence —
  `71c4898f7a7a306b0f6def06bda63d5192de456c8a849e1f08f83ba908cd3091`

The trainer additionally hashes this preregistration and the exact VG003,
VG009, and VG005 report/metadata/checkpoint files into epoch-zero state.

## Safety

This is offline recurrent supervised learning. Teacher plant actions,
FlightSim packets, sealed-test accesses, and Submission authorization are all
zero. The authoritative VQ2 executable remains untouched at
`C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`.
