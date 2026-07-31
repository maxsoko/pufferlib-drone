# VQ2 VG040 nine-source recurrent refit preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one epoch-resumable offline refit tagged
`vq2_vg040_variable_gate_nine_source_refit_001`. Initialize the full recurrent
actor from numerically admitted VG033 epoch 6. Parent checkpoint/report/
admission SHA-256 values are
`56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`.

VG039 completed once from source commit `bb53f682...` and admits `1,901,389`
fresh legal records. Ordered Gate-1/2/3/4/5 reach is
`512/487/146/45/3`; it contains
`1,022,122/158,365/26,044/240` phase-2/3/4/5 records. All 27 hard corpus
predicates pass. Report/metadata/state/admission SHA-256 values are
`a0bdcd0d...`/`89134752...`/`41702ed8...`/`e01eb2e2...`. Completed resume
exits zero, leaves terminal state unchanged, and reproduces the report
byte-for-byte.

VG039 crash/miss/timeout counts `32/77/401` characterize VG033's visited
failure distribution and do not relax deployment safety. This refit is
offline only.

## Fixed training contract

- Seed `429153`, six complete epochs, eight agents per source chunk,
  256-step recurrent BPTT, four optimizer exposures for every chunk containing
  a public-phase transition, AdamW learning rate `5e-6`, weight decay `1e-5`,
  gradient clip `1.0`, and smoothness weight `1e-4`.
- Initialize every actor parameter from VG033 epoch 6. Train the entire
  encoder, recurrent core, and four-channel decoder; do not reset recurrent
  weights or fit only the decoder.
- Every optimizer update independently normalizes and includes clean, VG009,
  recovered VG012, VG016, VG019, VG024, VG027, VG032, and VG039 sources with
  fixed weights `0.22/0.02/0.02/0.04/0.05/0.05/0.08/0.12/0.40`.
- Raw record counts never choose source importance. Validation agents are
  fixed independently within each source; VG032 and VG039 reserve 64 each.
- Bind `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=1` before first run and every
  resume.
- Actor input remains the legal 4,119-value ABI. Teacher plant actions,
  privileged actor input, sampling, clipping, fallback, FlightSim packets,
  and sealed-test access are zero.

## Numerical admission

Select the epoch with minimum fixed nine-source validation objective. Admit
only if it is a trained epoch, all metrics are finite, every epoch proves exact
source-weight sums and at least `4.0x` transition exposure, and:

- the selected nine-source objective strictly improves the VG033 baseline;
- VG039 weighted validation MSE strictly improves its VG033 baseline;
- clean/VG009/recovered-VG012/VG016/VG019/VG024/VG027/VG032/VG039 weighted
  MSE is at most `0.02/0.02/0.06/0.05/0.04/0.04/0.10/0.12/0.12`.

Numerical admission authorizes only a separately preregistered fresh paired
teacher-free diagnostic. It is not deployment admission.

## Source identity, resume, and remote gate

Before optimizer creation, bind the exact pushed Git commit, runtime, compiled
extension, parent checkpoint/report/admission, all nine dataset reports and
metadata, every prior admission/recovery artifact, VG039 admission, this
preregistration, runner, trainer/core/tests, model and dataset loaders, goal
prompt, configuration, thread limits, and zero-authority safety fields.

Write atomic state before the first update and after every epoch. Resume only
when every bound identity, RNG state, optimizer/scaler state, history, and
epoch counter matches. A completed or numerically rejected tag cannot train
again unchanged.

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, a fresh SM89 float32
vision build, both native regression suites, and the complete focused trainer
test shard.

Frozen new source surface:

- `scripts/train_vq2_variable_gate_nine_source_refit.py` — SHA-256
  `22682e75ea3274f603af04d1cbb9db436c3df1076a66f91af4a4ee7f0bacc134`
- `scripts/run_vq2_vg040_vast.sh` — SHA-256
  `386e660bc09f7782cdf82ddc59a4d0255235d71f13dbcaac2d36f299f15d8434`
- `tests/test_train_vq2_variable_gate_nine_source_refit.py` — SHA-256
  `15470bc9c06cc5952ccd35c45356bd4510f1d8a29c23622f6b0b936fa6f6f747`
- `scripts/train_vq2_variable_gate_eight_source_refit.py` — SHA-256
  `cb2f4287292b2d4e5c854e69f8118617d5e35378ec5e6c1237730566681f36b8`
- `scripts/train_vq2_variable_gate_seven_source_refit.py` — SHA-256
  `c76902c89714cc4a2c786d9beb5ecf7064f8f29947fd4ccc424c503d341126ec`
- `docs/vq2_vg039_variable_gate_dagger_round9_admission_2026-07-31.json` —
  SHA-256
  `e01eb2e28c65f827cfb39916c28d56320c57a4c23513ee61553447bd236b11cc`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, teacher plant actions,
and sealed-test accesses are zero. VQ2 Submission remains forbidden without
explicit user authorization.
