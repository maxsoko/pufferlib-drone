# VQ2 VG027 VG025-visited DAgger round 6 preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg027_variable_gate_dagger_round6_vg025_visited_512`. The plant actor is
the numerically admitted VG025 epoch-5 recurrent deterministic policy,
checkpoint/report/admission SHA-256 values
`85671c31...`/`b9d22d28...`/`633a34d5...`.

VG026 is the terminal teacher-free rejection for that actor. Bind aggregate
report and rejection-evidence SHA-256 values
`324634e2...`/`866e240c...`. Across 256 episodes it recorded zero finishes,
168 crashes (72 low and 96 lateral/XY), 88 misses, and zero timeouts. Ordered
Gate-1/Gate-2/Gate-3/Gate-4 reach was 256/97/2/0; by course count 5/8/11/12,
Gate-2 reach was 30/22/22/23 of 64 and Gate-3 reach was 2/0/0/0. This
authorizes oracle queries on VG025's newly reached Gate-2/Gate-3 and lateral-
recovery states; it authorizes neither an unchanged VG026 retry nor live
activity.

## Fixed collection contract

- Seed `429119`, 512 vector agents, exactly one terminal episode per agent,
  and exactly 64 episodes at each uniformly randomized gate count 5 through 12.
- Maximum 4,096 steps per episode at 64 Hz and at least 1,000,000 valid labels.
  The longer horizon is causal: VG026 count 5 averages 3,934 steps, reaches
  Gate 2 on 30/64 courses and Gate 3 on 2/64, while the old 1,024-step VG024
  corpus targeted only the initial Gate-1 transition.
- Require at least 95% ordered Gate-1 reach, 20% ordered Gate-2 reach, and at
  least one stored phase-2 record. These are corpus-usefulness predicates based
  on the source-locked VG026 frontier, not relaxed deployment safety.
- The plant receives only VG025's deterministic recurrent four-action output.
  Teacher blend/action emission, sampling, clipping, fallback, and student
  updates are zero.
- The SF016-admitted oracle is queried after each visited native state. Its
  action is stored as a training label only and never reaches the plant.
- Store only the unchanged legal 4,119-value actor ABI and causal executed
  action history. Privileged state and total gate count are absent.

Crash, miss, timeout, and crossing-margin rates describe the failure states
being labeled and remain diagnostics for corpus admission. Later teacher-free
screens retain zero-crash and reliable full-course requirements.

## Hard corpus admission

Admit only with exactly 512 terminal episodes, positive lengths no greater than
4,096, one final terminal marker per episode, label count equal to summed
lengths and at least 1,000,000, exact uniform course-count mass, the reach and
phase-2 predicates above, zero ordering/action/wire/thrust/non-finite/query/
action-history/public-phase fault, and the exact 17-value phase-record shape.

A failed predicate writes terminal rejection before raising. An admitted or
rejected VG027 tag cannot be rerun unchanged.

## Source identity and remote gate

Bind the pushed Git commit, runtime, compiled extension, generic collector,
VG027 wrapper/runner/preregistration, goal prompt, VG025 checkpoint/report/
admission, VG026 aggregate/rejection, SF016 oracle report, oracle/observation/
native sources and config, fixed collection contract, and zero-authority
safety fields before the first vector step. Resume is exact-identity only and
only before a terminal result.

Require the preserved Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 20 GiB free disk, Clang/OpenMP, ccache, a fresh architecture-
matched float32 native build, both native suites, and the complete focused
test shard.

Frozen new source surface:

- `scripts/collect_vq2_vg027_variable_gate_dagger.py` — `cc630062...`
- `scripts/run_vq2_vg027_vast.sh` — `1094f89f...`
- `tests/test_collect_vq2_vg027_variable_gate_dagger.py` — `b2e8d62d...`

## Safety and next authority

FlightSim packets, shadow, VQ2 Training, Submission, sealed N712 access,
teacher plant actions, and student updates are zero. Admission authorizes only
a separately preregistered source-balanced refit retaining every prior anchor.
