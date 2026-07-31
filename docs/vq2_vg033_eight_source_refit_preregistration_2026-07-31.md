# VQ2 VG033 eight-source recurrent refit preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one epoch-resumable offline refit tagged
`vq2_vg033_variable_gate_eight_source_refit_001`. Initialize the full
recurrent actor from numerically admitted VG028 epoch 6. Parent
checkpoint/report/admission SHA-256 values are
`ec1206b5...`/`b51b7a9b...`/`3282c98f...`.

VG032 completed once from source commit `04ee2edb...` and admits `1,636,103`
legal records. Ordered Gate-1/Gate-2/Gate-3/Gate-4 reach is
`512/438/67/19`; it contains `734,846` phase-2, `67,253` phase-3, and
`13,244` phase-4 records. All 24 hard corpus predicates pass. Report,
metadata, terminal-state, and admission SHA-256 values are
`f725afd9...`/`b4d43848...`/`b9a1a9fb...`/`7887a69a...`.
Completed resume exits zero, leaves terminal state unchanged, and reproduces
the report byte-for-byte.

VG032 crash/miss/timeout counts `59/214/239` describe VG028's visited failure
distribution and do not relax deployment safety. This refit is offline only.

## Fixed training contract

- Seed `429132`, six complete epochs, eight agents per source chunk,
  256-step recurrent BPTT, four optimizer exposures for every chunk
  containing a public-phase transition, AdamW learning rate `5e-6`, weight
  decay `1e-5`, gradient clip `1.0`, and smoothness weight `1e-4`.
- Initialize every actor parameter from VG028 epoch 6. Train the entire
  encoder, recurrent core, and four-channel decoder; do not reset recurrent
  weights or fit only the decoder.
- Every optimizer update independently normalizes and includes clean, VG009,
  recovered VG012, VG016, VG019, VG024, VG027, and VG032 sources with fixed
  weights `0.25/0.03/0.03/0.06/0.08/0.08/0.12/0.35`.
- Raw record counts never choose source importance. Validation agents are
  fixed independently within each source; VG032 reserves 64 agents.
- Actor input remains the legal 4,119-value ABI. Teacher plant actions,
  privileged actor input, sampling, clipping, fallback, FlightSim packets,
  and sealed-test access are zero.

## Numerical admission

Select the epoch with minimum fixed eight-source validation objective. Admit
only if it is a trained epoch, all metrics are finite, every epoch proves exact
source-weight sums and at least `4.0x` transition exposure, and:

- the selected eight-source objective strictly improves the VG028 baseline;
- VG032 weighted validation MSE strictly improves its VG028 baseline;
- clean/VG009/recovered-VG012/VG016/VG019/VG024/VG027/VG032 weighted MSE is
  at most `0.02/0.02/0.06/0.05/0.04/0.04/0.10/0.12`.

Numerical admission authorizes only a separately preregistered fresh paired
teacher-free diagnostic. It is not deployment admission.

## Source identity, resume, and remote gate

Before optimizer creation, bind the exact pushed Git commit, runtime, compiled
extension, parent checkpoint/report/admission, all eight dataset reports and
metadata, every prior admission/recovery artifact, VG032 admission, this
preregistration, runner, trainer/core/tests, model and dataset loaders, goal
prompt, configuration, and zero-authority safety fields.

Write atomic state before the first update and after every epoch. Resume only
when every bound identity, RNG state, optimizer/scaler state, history, and
epoch counter matches. A completed or numerically rejected tag cannot train
again unchanged.

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 20 GiB free disk, Clang/OpenMP, ccache, a fresh SM89 float32
vision build, both native regression suites, and the complete focused trainer
test shard.

Frozen new source surface:

- `scripts/train_vq2_variable_gate_seven_source_refit.py` — SHA-256
  `c76902c89714cc4a2c786d9beb5ecf7064f8f29947fd4ccc424c503d341126ec`
- `scripts/train_vq2_variable_gate_eight_source_refit.py` — SHA-256
  `cb2f4287292b2d4e5c854e69f8118617d5e35378ec5e6c1237730566681f36b8`
- `scripts/run_vq2_vg033_vast.sh` — SHA-256
  `3e85240b6e89346bcd1780f7c6eef606fbefd74f18eb25c001419edc26946ef8`
- `tests/test_train_vq2_variable_gate_seven_source_refit.py` — SHA-256
  `065f147cd45f5b2e93d8ad28e05279cc2ac6ace08f56d2abfa6c74f221f44a0e`
- `tests/test_train_vq2_variable_gate_eight_source_refit.py` — SHA-256
  `335a5e103ced6b024f1e08b6a827ce1d9df85b2063fee49ef45bed5fb7127a39`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, teacher plant actions,
and sealed-test accesses are zero. VQ2 Submission remains forbidden without
explicit user authorization.
