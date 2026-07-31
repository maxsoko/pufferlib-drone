# VQ2 VG032 VG028-visited higher-phase DAgger preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg032_variable_gate_dagger_round7_vg028_visited_512`. VG028 is the
sole plant actor. Its checkpoint, training report, and numerical-admission
SHA-256 values are `ec1206b5...`/`b51b7a9b...`/`3282c98f...`.

VG031 is terminally rejected and must never resume or retry unchanged. Its
source-bound early-rejection evidence SHA-256 is `aed9cc06...`. The first
64 count-5 episodes recorded 5 full finishes, 27 crashes, 30 misses, and 2
timeouts. Ordered Gate-1 through Gate-5 reach was `64/54/18/16/5`. With only
192 unrun episodes, its maximum possible 197 finishes was already below the
preregistered 231-finish floor; zero-crash admission was also impossible.
Transport, action history, public phase, ordering, and envelopes were clean.
This evidence authorizes a causally distinct collection at VG028-visited
higher-phase states, but no live action.

## Fixed collection contract

- Seed `429131`, 512 vector agents, exactly one terminal episode per agent,
  and exactly 64 uniformly randomized courses at each gate count 5 through 12.
- Maximum 4,096 steps per episode at 64 Hz and at least 1,000,000 valid labels.
- Require at least 95% ordered Gate-1 reach, 50% ordered Gate-2 reach, 10%
  ordered Gate-3 reach, and at least one stored phase-2 and phase-3 record.
  These corpus-usefulness floors are below VG031's observed `84.375%` Gate-2
  and `28.125%` Gate-3 count-5 reach while remaining materially positive.
- VG028's deterministic recurrent mean emits the complete four-channel plant
  action at every native step. Sampling, clipping, teacher blend/action,
  fallback, analytic override, checkpoint switching, and student updates are
  zero.
- Query the admitted SF016 alignment oracle after each VG028-visited native
  state. Store its action only as the training label; it never reaches the
  plant.
- Store only the unchanged legal 4,119-value actor ABI and causal executed
  action history. Privileged state and total gate count are absent.

Crash, miss, timeout, and crossing-margin rates are corpus diagnostics because
the purpose is to label VG028's actual later-course failure states. Later
teacher-free screens retain zero-crash and reliable full-course requirements.

## Hard corpus admission

Admit only with exactly 512 terminal episodes, positive lengths no greater
than 4,096, one final terminal marker per episode, label count equal to summed
lengths and at least 1,000,000, exact uniform course-count mass, all reach and
phase-record floors, and zero ordering/action/wire/thrust/non-finite/query/
action-history/public-phase fault. The phase-record vector must have exactly
17 entries.

A failed predicate writes terminal rejection before raising. An admitted or
rejected VG032 tag cannot be rerun unchanged.

## Source identity and remote gate

The manifest is
`docs/vq2_vg032_variable_gate_dagger_manifest_2026-07-31.json`, SHA-256
`d6da45a8cf38d8da2f47c8c025e81d23aee7b7dfec36e332fc6ef485c27b8c8d`.
Before the first vector step, bind the exact pushed Git commit, runtime,
compiled extension, collector, manifest, runner, this preregistration, goal
prompt, VG028 checkpoint/report/admission, VG031 rejection evidence, SF016
oracle report, legal observation/oracle/native sources and config, fixed
collection contract, and zero-authority safety fields. Resume is
exact-identity only and only before a terminal result.

Require the preserved Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 20 GiB free disk, Clang/OpenMP, ccache, a fresh
architecture-matched float32 native build, both native suites, and the focused
test shard.

Frozen new source surface:

- `scripts/collect_vq2_staged_variable_gate_dagger.py` — SHA-256
  `0428813e6fcca9d01a6e37183f7cd82739cad0eb458a20c0e2116a5d616e1523`
- `scripts/run_vq2_vg032_vast.sh` — SHA-256
  `e02784100442a35e1089d14957b9772253ba5e88fcb3dd3f0715fe786f892ba4`
- `tests/test_collect_vq2_staged_variable_gate_dagger.py` — SHA-256
  `b644e910b9d31d421fa87e31c74bc4ca0c862725a23da947317004051faa7b0e`
- `docs/vq2_vg031_variable_gate_teacher_free_early_rejection_2026-07-31.json`
  — SHA-256
  `aed9cc0651a204fe4d29ddbd6ea34d7d376865f38ae0d0aca1160efbaea1f244`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, sealed-test access,
teacher plant actions, and student updates are zero. Admission authorizes only
a separately preregistered source-balanced recurrent refit retaining every
prior anchor. VQ2 Submission remains forbidden without explicit user
authorization.
