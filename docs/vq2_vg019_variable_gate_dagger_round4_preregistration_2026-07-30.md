# VQ2 VG019 VG017-visited DAgger round 4 preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg019_variable_gate_dagger_round4_vg017_visited_512`. The plant actor is
the admitted VG017 epoch-11 deterministic recurrent policy, SHA-256
`6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8`.
Its training report SHA-256 is
`476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b`
and its independent admission evidence SHA-256 is
`df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6`.

VG018 is terminal rejection evidence for that actor: `0/256` finishes,
`254/256` Gate-1 reaches, `4/256` Gate-2 reaches, one Gate-3 reach,
`168/256` crashes, `88/256` misses, and zero timeout, action-parity,
public-phase, non-finite, ordering, or action-envelope fault. Its aggregate
report SHA-256 is
`40be8bf0041e9b183c5ad057ae6b4e5a20eb82ef21a972a8f0390e9d3c65ca09`
and rejection-evidence SHA-256 is
`b0d7ddb33b887cdfeb840710915ebb07ff9ec8455d8e1890c1be722db6e8af36`.
This authorizes fresh oracle queries at VG017's newly visited Gate-2
transition states; it does not authorize retrying VG018 or live activity.

## Fixed collection contract

- Seed `429093`, 512 vector agents, and exactly one terminal episode per agent.
- Uniform randomized gate counts 5 through 12: exactly 64 episodes per count.
- Maximum 1,024 steps per episode at 64 Hz and at least 300,000 valid records.
- At least 95% ordered Gate-1 reach and at least one stored post-Gate-1 record.
- The 1,024-step horizon retains the first 16 seconds, including the solved
  Gate-1 prefix and failing Gate-2 response, without accumulating long
  post-miss trajectories. The lower record floor reflects VG018's higher rate
  of terminal low crashes; it remains a large, transition-dense corpus.
- The plant receives only the deterministic mean of VG017's complete
  four-action recurrent output. Teacher blend, analytic action emission,
  sampling, clipping, fallback, and student updates are all zero.
- The SF016-admitted alignment oracle is queried after observing each visited
  native state. Its action is a training label only and never reaches the
  plant.
- Each stored student observation is the unchanged 4,118-value legal
  camera/IMU/actuator/action/timing ABI plus one causal held 4 Hz public phase
  encoded as `clamp(active_gate_index,0,16)/16`. Privileged state and total
  gate count are never stored in the student observation.

VG018 proves that crash and low later-gate reach are properties of the visited
distribution that need labels. Crash rate, Gate-2 reach, missed-gate rate, and
timeout rate are therefore collection diagnostics, not VG019 corpus-admission
predicates. This does not relax the later teacher-free policy screen, which
still requires zero crash and reliable full-course completion.

## Hard corpus admission

The collection is admitted only when all of these predicates hold:

- exactly 512 native terminal episodes with positive length at most 1,024;
- exactly one terminal marker per episode, at its final valid record;
- valid-label count equals the sum of episode lengths and is at least 300,000;
- exactly uniform gate-count mass for counts 5 through 12;
- at least 95% ordered Gate-1 reach and nonzero phase-1 records;
- zero out-of-order, action-envelope, wire-rate-envelope, and thrust-envelope
  metric;
- zero non-finite or out-of-envelope oracle query label;
- executed student-action history error at most `1e-7`;
- zero public-phase off-tick change, decrease, and skip, with raw encoding
  error at most `1e-6`; and
- the phase-record vector has exactly 17 entries.

Crossing-margin violation is also diagnostic-only at its `0.50 m` diagnostic
radius; the official aperture remains `0.75 m`. A failed hard predicate writes
a complete sibling rejection report and terminal state before raising. An
admitted or rejected VG019 tag cannot be rerun unchanged.

## Source identity and resume

Before the first vector step, state binds the Git commit, generic collector,
VG019 wrapper and runner, this preregistration, persistent goal prompt, VG017
checkpoint/report/admission, VG018 aggregate/rejection evidence, SF016 report,
oracle and observation/phase sources, compiled extension, runtime, fixed
collection contract, and zero-authority safety fields. Resume is allowed only
from exact identity and only before a terminal admitted/rejected result.

The remote bootstrap requires the exact pushed commit, a clean tracked tree,
the source-locked virtual environment, CUDA, at least 32 visible CPUs,
approximately 64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, and a fresh
float32 `drone_race_vision` build. Both native regression suites and focused
Python tests must pass before the first plant action.

## Frozen source surface

- `scripts/collect_vq2_variable_gate_dagger.py` —
  `cf0b8f6d0fba947b53fa82c7e06b848cb74e295fa578741bb9f66df47438b1d2`
- `scripts/collect_vq2_vg019_variable_gate_dagger.py` —
  `61b6e59473fba574d7592dd6b3ade0b5c1f67b3f3c752d5e56e683c52641ea78`
- `scripts/run_vq2_vg019_vast.sh` —
  `b838bcd724addcd58dbc541677e1593409c22fc8e1d6ffc19cf6aab16704c660`
- `tests/test_collect_vq2_variable_gate_dagger.py` —
  `e41bec07b2a892211c2804288644b9c30135065ef0fa94842de39d97aed58429`
- `tests/test_collect_vq2_vg019_variable_gate_dagger.py` —
  `03b6a5c57d6d472e073647b35462d50e9be73dab209bf181ee3e3b39cbb76f3f`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` —
  `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `scripts/eval_vq2_variable_gate_recurrent_policy.py` —
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `pufferlib/vq2_oracle.py` —
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
- `pufferlib/vq2_public_phase.py` —
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent.py` —
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` —
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md` —
  `052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`

The source identity also binds this preregistration, exact VG017/VG018
artifacts and evidence, SF016 report SHA
`a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`,
the native sources/config, and compiled extension before the first action.

## Safety and next authority

This is offline native DAgger data collection. FlightSim packets, Windows
shadow, bounded Training attempts, Submission selection, sealed N712 accesses,
teacher plant actions, and student updates are all zero. If admitted, VG019
authorizes only a separately preregistered five-source refit that preserves
VG003, VG009, recovered VG012, and VG016 while adding VG019 visited-state
labels. It provides no live-flight authority.
