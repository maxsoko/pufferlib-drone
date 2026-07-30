# VQ2 VG024 VG022-visited DAgger round 5 preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg024_variable_gate_dagger_round5_vg022_visited_512`. The plant actor is
the admitted VG022 epoch-9 deterministic recurrent policy, checkpoint SHA-256
`bd3f93d4...`; its training report/admission SHA-256 values are
`23d28182...`/`ac93a00d...`.

VG023 is terminal rejection evidence for that actor: `0/256` finishes,
`255/256` Gate-1 reaches, `26/256` Gate-2 reaches, zero Gate-3 reach,
`193/256` crashes, and `63/256` misses. Relative to VG018, Gate-2 reach rises
from 4 to 26, but crashes rise by 25 and shift from predominantly low to 101
lateral/XY crashes. Aggregate report/rejection-evidence SHA-256 values are
`2a3023fd...`/`88f403f0...`. This authorizes new oracle queries on VG022's
higher-turn and lateral-recovery states; it authorizes neither a VG023 retry
nor live activity.

## Fixed collection contract

- Seed `429109`, 512 vector agents, and exactly one terminal episode per agent.
- Uniform randomized gate counts 5 through 12: exactly 64 episodes per count.
- Maximum 1,024 steps per episode at 64 Hz and at least 300,000 valid records.
- At least 95% ordered Gate-1 reach and at least one stored post-Gate-1 record.
- The 1,024-step horizon retains the first 16 seconds around the causal
  Gate-1-to-Gate-2 turn/recovery while excluding the screen's long terminal
  proof horizon.
- The plant receives only VG022's deterministic full four-action recurrent
  output. Teacher blend/action emission, sampling, clipping, fallback, and
  student updates are zero.
- The SF016-admitted alignment oracle is queried after each visited native
  state. Its action is a training label only and never reaches the plant.
- Stored actor input is only the unchanged 4,118-value legal ABI plus the
  causal held 4 Hz public phase encoded as `clamp(index,0,16)/16`; privileged
  state and total gate count are absent.

VG023 proves that crash rate and later-gate reach are properties of the failure
distribution being labeled. They remain diagnostics for corpus admission, not
transport relaxations. Later teacher-free screens still require zero crash and
reliable full-course completion.

## Hard corpus admission

Admit only when all predicates hold:

- exactly 512 terminal episodes, each with positive length at most 1,024;
- one terminal marker per episode at its final valid record;
- valid-label count equals summed lengths and is at least 300,000;
- exact uniform count mass for counts 5 through 12;
- at least 95% ordered Gate-1 reach and nonzero phase-1 records;
- zero ordering, action/wire/thrust envelope, non-finite-label, action-history,
  off-tick phase, phase-decrease, phase-skip, or raw phase-encoding fault; and
- the phase-record vector has exactly 17 entries.

Crossing-margin violation remains diagnostic at `0.50 m`; the true aperture is
`0.75 m`. A failed hard predicate writes a complete terminal rejection before
raising. An admitted or rejected VG024 tag cannot be rerun unchanged.

## Source identity and resume

Before the first vector step, bind the Git commit, generic collector, VG024
wrapper/runner/preregistration, goal prompt, VG022 checkpoint/report/admission,
VG023 aggregate/rejection evidence, SF016 report, oracle/observation sources,
native sources/config, compiled extension, runtime, collection contract, and
zero-authority safety fields. Resume is exact-identity only and only before a
terminal result.

The remote bootstrap requires the exact pushed commit, clean worktree,
preserved virtual environment, CUDA, at least 32 CPUs, about 64 GiB RAM,
15 GiB free disk, Clang/OpenMP, ccache, and a fresh float32 native build. Both
native regression suites and focused tests must pass before the first action.

## Frozen source surface

- `scripts/collect_vq2_variable_gate_dagger.py` —
  `cf0b8f6d0fba947b53fa82c7e06b848cb74e295fa578741bb9f66df47438b1d2`
- `scripts/collect_vq2_vg024_variable_gate_dagger.py` —
  `6b8063bb6711e7d3dfb3dc1f968513f07c0df81bd798d5f64c3aabe5fa73068c`
- `scripts/run_vq2_vg024_vast.sh` —
  `574ad9acb0d06099eb2f3015cee5b93c046a2badc121dcb68a36fee9422d8942`
- `tests/test_collect_vq2_vg024_variable_gate_dagger.py` —
  `caf21cb66245739354b326bddec9aa708ee4458235bad323a916bd402c609c3e`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` —
  `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `pufferlib/vq2_oracle.py` —
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
- `pufferlib/vq2_public_phase.py` —
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent_phase.py` —
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md` —
  `052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`

The source identity also binds this preregistration and exact VG022/VG023/
SF016 evidence.

## Safety and next authority

This is offline native DAgger data collection. FlightSim packets, Windows
shadow, bounded Training attempts, Submission selection, sealed N712 accesses,
teacher plant actions, and student updates are all zero. If admitted, VG024
authorizes only a separately preregistered source-balanced refit retaining
VG003, VG009, recovered VG012, VG016, and VG019 while adding VG024. It grants no
live-flight authority.
