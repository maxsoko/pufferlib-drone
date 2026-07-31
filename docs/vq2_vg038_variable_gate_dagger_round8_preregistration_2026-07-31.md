# VQ2 VG038 VG033-visited higher-phase DAgger preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg038_variable_gate_dagger_round8_vg033_visited_512`. VG033 is the sole
plant actor. Its checkpoint, training report, and numerical-admission SHA-256
values are `56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`.

VG037 is terminally rejected and must never resume or retry unchanged. Its
source-bound early-rejection evidence SHA-256 is `bb14e1e1...`. Across its
completed fresh count-5 and count-8 prefix, 128 episodes produced 14 full
finishes, 45 crashes, 47 misses, and 22 timeouts. Ordered Gate-1 through
Gate-8 reach was `128/126/85/71/31/16/6/3`. Even 128 perfect unrun episodes
could produce at most 142 finishes, below the fixed 231-finish floor, while
zero-crash admission was already impossible. Transport, action history,
public phase, ordering, and envelopes were exact.

This is materially stronger later-phase reach than VG031 and authorizes one
causally distinct collection at VG033-visited Gate-3--8 states. It authorizes
no live action.

## Fixed collection contract

- Seed `429151`, 512 vector agents, exactly one terminal episode per agent,
  and exactly 64 uniformly randomized courses at each gate count 5 through 12.
- Maximum 4,096 steps per episode at 64 Hz and at least 1,000,000 valid labels.
- Require at least 95% ordered Gate-1 reach, 90% Gate-2 reach, 40% Gate-3
  reach, and 20% Gate-4 reach. Require at least one stored record in each
  public phase 2, 3, 4, and 5.
- VG033's deterministic recurrent mean emits the complete four-channel plant
  action at every native step. Sampling, clipping, teacher blend/action,
  fallback, analytic override, checkpoint switching, and student updates are
  zero.
- Query the admitted SF016 alignment oracle after each VG033-visited native
  state. Store its action only as the training label; it never reaches the
  plant.
- Store only the unchanged legal 4,119-value actor ABI and causal executed
  action history. Privileged state and total gate count are absent.
- Bind `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=1` before first run and resume
  to avoid VG036's host bookkeeping oversubscription.

Crash, miss, timeout, and crossing-margin rates are corpus diagnostics because
the purpose is to label VG033's actual later-course failure states. Later
teacher-free screens retain zero-crash and reliable full-course requirements.

## Hard corpus admission

Admit only with exactly 512 terminal episodes, positive lengths no greater
than 4,096, one final terminal marker per episode, label count equal to summed
lengths and at least 1,000,000, exact uniform course-count mass, all reach and
phase-record floors, and zero ordering/action/wire/thrust/non-finite/query/
action-history/public-phase fault. The phase-record vector must have exactly
17 entries.

A failed predicate writes terminal rejection before raising. An admitted or
rejected VG038 tag cannot be rerun unchanged.

## Source identity and remote gate

The manifest is
`docs/vq2_vg038_variable_gate_dagger_manifest_2026-07-31.json`, SHA-256
`63012cd210dda9ded5cacaa92103ef2eec6f303c89f4a9d4a695f54e7da3fcd2`.
Before the first vector step, bind the exact pushed Git commit, runtime,
compiled extension, generic and VG038 collectors, manifest, runner, this
preregistration, goal prompt, VG033 checkpoint/report/admission, VG037
rejection evidence, SF016 oracle report, legal observation/oracle/native
sources and config, fixed collection contract, and zero-authority safety
fields. Resume is exact-identity only and only before a terminal result.

Require the preserved Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 20 GiB free disk, Clang/OpenMP, ccache, a fresh
architecture-matched float32 native build, both native suites, and the focused
test shard.

Frozen new source surface:

- `scripts/collect_vq2_vg038_variable_gate_dagger.py` — SHA-256
  `c8b3b9ee37b206557cd72a6aac2ec19301acd445ad39aa73b7803363fdf1bbec`
- `scripts/run_vq2_vg038_vast.sh` — SHA-256
  `ab184a7dec78256ce580b0e4c8d1dd9e452dea746eb4eb0ae605f2d7bc4116f0`
- `tests/test_collect_vq2_vg038_variable_gate_dagger.py` — SHA-256
  `8015f08eaed194b7a7ab5eaf3dfc7a6d19972898e771c6f6f73bd89a4dc5c41e`
- `scripts/collect_vq2_staged_variable_gate_dagger.py` — SHA-256
  `0428813e6fcca9d01a6e37183f7cd82739cad0eb458a20c0e2116a5d616e1523`
- `scripts/collect_vq2_variable_gate_dagger.py` — SHA-256
  `cf0b8f6d0fba947b53fa82c7e06b848cb74e295fa578741bb9f66df47438b1d2`
- VG037 early-rejection evidence — SHA-256
  `bb14e1e1ff96c4bf13ca216b19ac08f72d7e704dd76a9b06867b3cdfbe6cb2a5`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, sealed-test access,
teacher plant actions, and student updates are zero. Admission authorizes only
a separately preregistered source-balanced recurrent refit retaining every
prior anchor. VQ2 Submission remains forbidden without explicit user
authorization.
