# VQ2 VG039 calibrated VG033-visited DAgger preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg039_variable_gate_dagger_round9_vg033_visited_512`. VG033 remains the
sole plant actor; its checkpoint/report/admission SHA-256 values remain
`56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`.

VG038 is terminally rejected and its arrays are quarantined. It cannot resume
or retry unchanged. The collection completed 512 fresh uniform count-5--12
episodes and produced 1,922,029 legal labels in `81.327282 s`. Every layout,
finite-action, query, action-history, public-phase, ordering, envelope, and
transport predicate passed. Phase 2/3/4/5 contributed
`1,061,932/139,254/28,334/272` records.

Only two preregistered corpus-usefulness floors failed: observed Gate-3/Gate-4
reach was `26.5625%/10.546875%` against `40%/20%`. The source-locked VG038
rejection report SHA-256 is
`7f91cb845b53736541d89aa836f2b1c079ac0944eaf6720b1ef444605792815d`.
VG039 uses a fresh seed and evidence-calibrated floors; no VG038 array is read.

## Fixed collection contract

- Seed `429152`, 512 vector agents, exactly one terminal episode per agent,
  and exactly 64 uniformly randomized courses at each gate count 5 through 12.
- Maximum 4,096 steps per episode at 64 Hz and at least 1,000,000 valid labels.
- Require at least 95% ordered Gate-1 reach, 90% Gate-2 reach, 20% Gate-3
  reach, and 5% Gate-4 reach. Require at least one stored record in each public
  phase 2, 3, 4, and 5.
- VG033's deterministic recurrent mean emits the complete four-channel plant
  action at every native step. Sampling, clipping, teacher blend/action,
  fallback, analytic override, checkpoint switching, and student updates are
  zero.
- Query SF016 after each fresh VG033-visited state and store the oracle action
  only as a training label. Store only the unchanged legal 4,119-value actor
  ABI plus causal executed action history.
- Bind `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=1` before first run and resume.

Crash, miss, timeout, and crossing-margin rates remain corpus diagnostics.
Later deterministic screens retain zero-crash and full-course reliability.

## Hard corpus admission

Admit only with exactly 512 terminal episodes, positive lengths no greater
than 4,096, one final terminal marker per episode, label count equal to summed
lengths and at least 1,000,000, exact uniform course-count mass, all reach and
phase-record floors, and zero ordering/action/wire/thrust/non-finite/query/
action-history/public-phase fault. The phase-record vector must have 17
entries.

A failed predicate writes terminal rejection before raising. An admitted or
rejected VG039 tag cannot be rerun unchanged.

## Source identity and remote gate

The manifest is
`docs/vq2_vg039_variable_gate_dagger_manifest_2026-07-31.json`, SHA-256
`900db1bdfbccc848f4ed3da5d48484a05fb03c99d900c7abe154f1ebd702846d`.
Before the first vector step, bind the exact pushed commit, runtime, compiled
extension, generic/VG038/VG039 collectors, manifest, runner, this
preregistration, goal prompt, VG033 checkpoint/report/admission, VG037 screen
evidence, VG038 rejection report, SF016 oracle report, legal/native sources
and config, fixed contract, and zero-authority fields. Resume is exact-identity
only before a terminal result.

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 20 GiB free disk, Clang/OpenMP, ccache, a fresh SM89 float32
native build, both native suites, and the focused test shard.

Frozen new source surface:

- `scripts/collect_vq2_vg039_variable_gate_dagger.py` — SHA-256
  `4eb20af406aaeb651f24e2fe34c30399de321f65bdde05cde9c5c9975ee11f03`
- `scripts/run_vq2_vg039_vast.sh` — SHA-256
  `eadb9181289723efde3ff330ec05eff40f13a16844700c6fa8828a0acc6f4e7e`
- `tests/test_collect_vq2_vg039_variable_gate_dagger.py` — SHA-256
  `d2c12dbff7747e43d9c029cf6b3227a2a9eda50f4389afdca2fe9920e882f202`
- `logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg038_variable_gate_dagger_round8_vg033_visited_512_rejection_report.json`
  — SHA-256
  `7f91cb845b53736541d89aa836f2b1c079ac0944eaf6720b1ef444605792815d`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, sealed-test access,
teacher plant actions, and student updates are zero. Admission authorizes only
a separately preregistered source-balanced recurrent refit retaining every
admitted prior anchor. VQ2 Submission remains forbidden without explicit user
authorization.
