# VQ2 VG025 six-source recurrent refit preregistration — 2026-07-30

## Evidence and one-run decision

Run one epoch-resumable offline refit tagged
`vq2_vg025_variable_gate_six_source_refit_001`. Start from the numerically
admitted VG022 epoch-9 recurrent checkpoint, SHA-256 `bd3f93d4...`. VG023 then
rejected this actor at `0/256`, with `26` Gate-2 reaches and `193` crashes.

VG024 is the causal successor corpus: 512 VG022-visited episodes, 512,351 legal
oracle labels, 510 Gate-1 reaches, 12 Gate-2 reaches, and 375,514 phase-1
records. All 20 hard corpus predicates pass. Report/metadata/admission SHA-256
values are `5d36213f...`/`dfe64745...`/`3b31cb2e...`. Completed-resume leaves
the terminal state unchanged. This authorizes the refit below and no live
activity.

## Fixed training contract

- Seed `429110`; initialize all model weights from VG022 epoch 9 and initialize
  one fresh AdamW optimizer.
- Exactly six epochs, learning rate `5e-6`, weight decay `1e-5`, gradient clip
  `1.0`, float16 CUDA autocast, deterministic CUDA settings, and float32 native
  dataset decode.
- BPTT chunks contain 256 recurrent steps. Any chunk containing a public phase
  transition receives exactly four optimizer exposures.
- Every optimizer update independently normalizes and includes all six sources.
  Raw record counts never set source weights.
- Fixed per-update weights are clean/VG009/recovered-VG012/VG016/VG019/VG024
  = `0.30/0.05/0.05/0.10/0.15/0.35`.
- Use four training agents per source per paired chunk. Hold out 32 clean agents
  and 64 agents from each DAgger source for deterministic validation.
- The action objective weights roll/pitch/thrust/yaw as `1/1/4/1`; temporal
  smoothness weight is `1e-4`.
- Save a source/runtime/RNG/optimizer-bound state atomically at epoch zero and
  after every epoch. Resume only that exact active state. A completed state is
  immutable and verifies its report/checkpoint before returning.

## Numerical selection and admission

Select the epoch with the lowest fixed source-balanced validation MSE. Numerical
admission requires all of the following:

- selected epoch is 1 through 6 and improves on VG022's six-source baseline;
- selected VG024 validation MSE strictly improves on its baseline and is at
  most `0.10`;
- clean/VG009/recovered-VG012/VG016/VG019 validation MSE is at most
  `0.02/0.02/0.06/0.05/0.04`, respectively;
- every epoch proves exact fixed source-weight accumulation and at least four
  transition exposures; and
- all values are finite.

Failure of numerical admission rejects VG025 for screening. Numerical admission
authorizes only a separately preregistered fresh teacher-free native screen; it
does not authorize shadow, FlightSim Training, or Submission.

## Remote source and compute gate

Before epoch zero, require the exact pushed commit and clean tracked worktree,
the preserved virtual environment, CUDA, at least 32 visible CPUs, about 64 GiB
RAM, 15 GiB free disk, Clang/OpenMP, ccache, and a fresh architecture-matched
float32 native build. Both native regression suites and the complete focused
training tests must pass.

Bind the goal prompt; this preregistration; runner and trainer; all recurrent,
dataset, and loss sources; all six report/metadata pairs; all five available
DAgger admissions; the recovered VG012 evidence; VG022 parent checkpoint,
report, and admission; the compiled extension; Git commit; and runtime manifest.

## Frozen new source surface

- `scripts/train_vq2_variable_gate_six_source_refit.py` —
  `a3142bf3f5125d3ab88850fdf94d2b23847822041f1f3fa8801903ac9f821b08`
- `scripts/run_vq2_vg025_vast.sh` —
  `9dc71bb9d9352b6594098925517d27d8100d635f1b1fbf7477458ad11cb3e87f`
- `tests/test_train_vq2_variable_gate_six_source_refit.py` —
  `b25f18c1d376023c9249a86030df537bdca82b7b69aa891f656511f0b9d37ede`

## Safety

The actor input remains the 4,118-value camera/IMU/action-history ABI plus one
held public-progress scalar. Privileged actor inputs, teacher blending or plant
actions, student action emission into any simulator, FlightSim packets, sealed
test accesses, shadow, Training, and Submission authority are all zero.
