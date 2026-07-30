# VQ2 VG012 VG010-visited DAgger round 2 preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline collection tagged
`vq2_vg012_variable_gate_dagger_round2_vg010_visited_512`. The plant actor is
the admitted VG010 epoch-12 deterministic recurrent policy, SHA-256
`a093475371a94bc284310bbe64609fe515a35e50be0e68a7fdfb781b55f03822`.
Its training report SHA-256 is
`dfe943b593f99c4ccfa53b04290a30cd3a58ea9fefd11d9489786e064b8734a5`
and its independent admission evidence SHA-256 is
`c0fbfc72c0c609b137d7240aa8347baa6f59926e2c26e137144d363e9e820850`.

VG011 is terminal rejection evidence for that actor: `0/256` finishes,
`176/256` Gate-1 reaches, zero Gate-2 reaches, `193/256` crashes, `63/256`
misses, and zero timeout, action-parity, public-phase, non-finite, or action
envelope fault. Its aggregate report SHA-256 is
`3e470b1896881eaab74aa1391fdb51c3639d6ed1149e0678b3d4e35cb0e7cd3d`
and rejection-evidence SHA-256 is
`53d5ed65b680b83c278993c2517553043de1444460e8780ad2ddb8f0a61271f5`.
The fresh screen predicts about `188,820` records for 512 mixed-count
episodes. This evidence authorizes collecting labels at the failed actor's
visited states; it does not authorize retrying VG011 or live activity.

## Fixed collection contract

- Seed `429064`, 512 vector agents, and exactly one terminal episode per agent.
- Uniform randomized gate counts 5 through 12: exactly 64 episodes per count.
- Maximum 2,048 steps per episode at 64 Hz and at least 150,000 valid records.
- At least 60% ordered Gate-1 reach and at least one stored post-Gate-1 record.
- The plant receives only the deterministic mean of VG010's complete
  four-action recurrent output. Teacher blend, analytic action emission,
  sampling, clipping, fallback, and student updates are all zero.
- The SF016-admitted alignment oracle is queried after observing each visited
  native state. Its action is a training label only and never reaches the
  plant.
- Each stored student observation is the unchanged 4,118-value legal
  camera/IMU/actuator/action/timing ABI plus one causal held 4 Hz public phase
  encoded as `clamp(active_gate_index,0,16)/16`. Privileged state and total
  gate count are never stored in the student observation.

VG011 shows that crashes and zero Gate-2 reach are properties of the
distribution that must be labeled. Consequently, crash rate and Gate-2 reach
are recorded as diagnostics and are not VG012 corpus-admission predicates.
This is a deliberate change from VG009's 5% crash and 1% Gate-2 collection
gates; keeping those gates would reject the exact failure states needed for
the second DAgger round. It does not weaken any later teacher-free policy
admission screen, which still requires zero crash and reliable full-course
completion.

## Hard corpus admission

The collection is admitted only when all of these predicates hold:

- exactly 512 native terminal episodes with positive length at most 2,048;
- exactly one terminal marker per episode, at its final valid record;
- valid-label count equals the sum of episode lengths and is at least 150,000;
- exactly uniform gate-count mass for counts 5 through 12;
- at least 60% ordered Gate-1 reach and nonzero phase-1 records;
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
rejected VG012 tag cannot be rerun unchanged.

## Source identity and resume

Before the first vector step, state binds the Git commit, generic collector,
VG012 wrapper and runner, this preregistration, persistent goal prompt, VG010
checkpoint/report/admission, VG011 aggregate/rejection evidence, SF016 report,
oracle and observation/phase sources, compiled extension, runtime, fixed
collection contract, and zero-authority safety fields. Resume is allowed only
from exact identity and only before a terminal admitted/rejected result.

The remote bootstrap requires the exact pushed commit, a clean tracked tree,
the existing source-locked virtual environment, CUDA, at least 32 visible
CPUs, 15 GiB free disk, Clang/OpenMP, ccache, and a fresh float32
`drone_race_vision` build. Both native regression suites and focused Python
tests must pass before the first plant action.

## Safety and next authority

This is offline native DAgger data collection. FlightSim packets, Windows
shadow, bounded Training attempts, Submission selection, sealed N712 accesses,
teacher plant actions, and student updates are all zero. If admitted, VG012
authorizes only a separately preregistered three-source refit that preserves
VG003 clean anchors and VG009 failure anchors while adding the new VG012
visited-state labels. It provides no live-flight authority.
