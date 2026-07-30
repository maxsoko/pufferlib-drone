# VQ2 VG016 VG014-visited DAgger round 3 preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg016_variable_gate_dagger_round3_vg014_visited_512`. The plant actor is
the admitted VG014 epoch-8 deterministic recurrent policy, SHA-256
`60d69a1f80a1551b190cd4b517dfa029bc3dbe48728a853a47b658df0866d79e`.
Its training report SHA-256 is
`067d9c198c14959dbb7221d34f9fec0f065a3cee41043e3ce0ff1f9a7bbf0a16`
and its independent admission evidence SHA-256 is
`073fc31e143e2d8e61e40cead685698c47e012b7153b32622bd4dc202fbcfa72`.

VG015 is terminal rejection evidence for that actor: `0/256` finishes,
`246/256` Gate-1 reaches, `2/256` Gate-2 reaches, zero Gate-3 reaches,
`137/256` crashes, `119/256` misses, and zero timeout, action-parity,
public-phase, non-finite, ordering, or action-envelope fault. Its aggregate
report SHA-256 is
`bb655666484125a46ce23612c2bc4cb4a18295991c10654b94a7e9c0be782cda`
and rejection-evidence SHA-256 is
`568f948e861cef0634d7e1464c8d7869d1767eb77a1829b380a529036c1cbde6`.
This evidence authorizes collecting labels at the failed actor's Gate-2
transition states; it does not authorize retrying VG015 or live activity.

## Fixed collection contract

- Seed `429079`, 512 vector agents, and exactly one terminal episode per agent.
- Uniform randomized gate counts 5 through 12: exactly 64 episodes per count.
- Maximum 1,024 steps per episode at 64 Hz and at least 350,000 valid records.
- At least 90% ordered Gate-1 reach and at least one stored post-Gate-1 record.
- The 1,024-step horizon deliberately concentrates this round on the first
  16 seconds. VG015 passes Gate 1 around step 233, then spends most remaining
  time in the failing Gate-2 phase; truncating the old 2,048-step horizon
  prevents long post-miss divergence from dominating the labels.
- The plant receives only the deterministic mean of VG014's complete
  four-action recurrent output. Teacher blend, analytic action emission,
  sampling, clipping, fallback, and student updates are all zero.
- The SF016-admitted alignment oracle is queried after observing each visited
  native state. Its action is a training label only and never reaches the
  plant.
- Each stored student observation is the unchanged 4,118-value legal
  camera/IMU/actuator/action/timing ABI plus one causal held 4 Hz public phase
  encoded as `clamp(active_gate_index,0,16)/16`. Privileged state and total
  gate count are never stored in the student observation.

VG015 shows that crashes and near-zero Gate-2 reach are properties of the
distribution that must be labeled. Consequently, crash rate and Gate-2 reach
are diagnostics and are not VG016 corpus-admission predicates. The shortened
horizon will turn surviving long misses into timeouts; timeout is likewise a
collection diagnostic. None of these choices weakens a later teacher-free
policy screen, which still requires zero crash and reliable full-course
completion.

## Hard corpus admission

The collection is admitted only when all of these predicates hold:

- exactly 512 native terminal episodes with positive length at most 1,024;
- exactly one terminal marker per episode, at its final valid record;
- valid-label count equals the sum of episode lengths and is at least 350,000;
- exactly uniform gate-count mass for counts 5 through 12;
- at least 90% ordered Gate-1 reach and nonzero phase-1 records;
- zero out-of-order, action-envelope, wire-rate-envelope, and thrust-envelope
  metric;
- zero non-finite or out-of-envelope oracle query label;
- executed student-action history error at most `1e-7`;
- zero public-phase off-tick change, decrease, and skip, with raw encoding
  error at most `1e-6`; and
- the phase-record vector has exactly 17 entries.

Crossing-margin violation, crash rate, Gate-2 reach, missed-gate rate, and
timeout rate are diagnostics. A failed hard predicate writes a complete
sibling rejection report and terminal state before raising. An admitted or
rejected VG016 tag cannot be rerun unchanged.

## Source identity and resume

Before the first vector step, state binds the Git commit, generic collector,
VG016 wrapper and runner, this preregistration, persistent goal prompt, VG014
checkpoint/report/admission, VG015 aggregate/rejection evidence, SF016 report,
oracle and observation/phase sources, compiled extension, runtime, fixed
collection contract, and zero-authority safety fields. Resume is allowed only
from exact identity and only before a terminal admitted/rejected result.

The remote bootstrap requires the exact pushed commit, a clean tracked tree,
the existing source-locked virtual environment, CUDA, at least 32 visible
CPUs, approximately 64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, and a
fresh float32 `drone_race_vision` build. Both native regression suites and
focused Python tests must pass before the first plant action.

## Frozen source surface

- `scripts/collect_vq2_variable_gate_dagger.py` —
  `cf0b8f6d0fba947b53fa82c7e06b848cb74e295fa578741bb9f66df47438b1d2`
- `scripts/collect_vq2_vg016_variable_gate_dagger.py` —
  `c988ed4846b58d68a12cb5b56fb8af1a21f7f62b579a7a6c6a89b4464f6021c1`
- `scripts/run_vq2_vg016_vast.sh` —
  `7991d1b159fab85f8202b1f830abafa413d1deac9baaa346081478031af645d6`
- `tests/test_collect_vq2_variable_gate_dagger.py` —
  `e41bec07b2a892211c2804288644b9c30135065ef0fa94842de39d97aed58429`
- `tests/test_collect_vq2_vg016_variable_gate_dagger.py` —
  `e1f79b791b623fee94c89c43a45ab802fb18a33e534f34f20aa3e0c6d45c3919`
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

The source identity also binds this preregistration, exact VG014/VG015
artifacts and evidence, the SF016 report SHA
`a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`,
the native sources/config, and compiled extension before the first action.

## Safety and next authority

This is offline native DAgger data collection. FlightSim packets, Windows
shadow, bounded Training attempts, Submission selection, sealed N712 accesses,
teacher plant actions, and student updates are all zero. If admitted, VG016
authorizes only a separately preregistered source-balanced refit that preserves
VG003, VG009, and recovered-VG012 while adding VG016 visited-state labels. It
provides no live-flight authority.
