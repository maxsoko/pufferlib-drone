# VQ2 VG007 variable-gate DAgger round 1 preregistration — 2026-07-29

## Trigger and purpose

VG006 rejected the sole VG005 checkpoint at `0/256`, but localizes the failure
to exactly the Gate-1 -> Gate-2 transition: 244/256 Gate-1 passes, 20/256
Gate-2 passes, no Gate-3 reach, and two crashes. This satisfies the goal
prompt's explicit condition to enter Stage-3 DAgger on the failed transition.
VG006 aggregate report SHA-256 is
`1f848400806e2ba8c353f557db1e30099ea2c570fbb6c16f0d7664c1c1f057f0`.

Collect one legal/public dataset tagged
`vq2_vg007_variable_gate_dagger_round1_256`. The VG005 deterministic mean
drives every plant step. The admitted Python alignment oracle queries the
training-only native state at each VG005-visited state and supplies labels but
never drives, blends, clips, or overrides the plant. The resulting failure
distribution will later be paired source-balanced with the immutable VG003
clean corpus; it never replaces or numerically dilutes that anchor.

## Fixed rollout and storage

- Seed `429047`; 256 fixed instances/episodes; exact-uniform construction-time
  gate counts 5 through 12 (32 each).
- Full starts and admitted randomized reachable geometry, true 0.75 m
  aperture. Collection is capped by native terminal at 2,048 policy steps
  (32 seconds), which spans the measured Gate-2 failures without allocating a
  misleading long post-failure tail.
- Teacher action blend, spline, minimum-jerk, alignment governor, roll-until,
  and action reward paths are all zero/disabled. Plant actions are only VG005
  checkpoint SHA
  `f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3`.
- Oracle query source SHA
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
  is frozen by SF016 parity report SHA
  `a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`.
- Each record stores only `mask[4096] uint8 + legal tail[22] float32 + held
  public phase[1] float32 + oracle action[4] float32`, plus terminal/valid
  flags. Phase is sampled at 4 Hz and encoded as official index divided by 16.
  No total count, vehicle/gate pose, oracle state, or other native privileged
  value is written.

The action label is the four-output oracle query at the current student state;
the next native action-history slot must match the VG005 action actually sent
within `1e-7`. Every instance has one contiguous valid prefix and exactly one
terminal marker on its last row.

## Admission

Require 256 native terminal logs, at least 100,000 records, exact 1/8 episode
mass at every count 5--12, at least 90% Gate-1 reach, at least 1% Gate-2 reach,
and at least one stored phase-1 record. Preserve expected failure outcomes:
crash is allowed only up to 5%; miss/timeout is allowed because these are the
states DAgger must repair. Out-of-order, crossing-margin, action, wire-rate,
and thrust-envelope violations remain zero.

Transport must have finite/enveloped student and query actions, executed-
action error at most `1e-7`, exact `/16` encoding, and zero phase off-tick
change/decrease/skip. Metadata and report record per-phase counts, student vs.
oracle MSE/MAE and ranges, native gate metrics, array hashes, source/runtime,
and zero persisted privilege/total-count values.

## Resume and rejection

Before rollout, sibling state locks source commit/hashes, runtime, extension,
seed, dimensions, cap, and safety. An interruption may clear only this tag's
partial output/staging and deterministically recollect with exact identity.
Once state exists, the Vast wrapper must not rebuild, test, or install. An
admission failure writes terminal `rejected` and forbids unchanged retry; a
successful finalization writes terminal `admitted`, manifest and report hash.

## Frozen sources

- `scripts/collect_vq2_variable_gate_dagger.py` — `c36c083b2c82be7752e03d823c10a84affd78f515fe3eb631324d36f6e8f4940`
- `scripts/run_vq2_vg007_vast.sh` — `cd205ef9978335c68d8c3a19ddb1ebe7f648a181a6175a354b5414ef7dc0edf5`
- `tests/test_collect_vq2_variable_gate_dagger.py` — `e9f9a035a91a3a78454334b57f89b67aed20d31c650ef8f6077afef551a69066`
- `pufferlib/vq2_oracle.py` — `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_public_phase.py` — `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` — `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `scripts/eval_vq2_variable_gate_recurrent_policy.py` — `e4ca39c4f66ef2348c123f9d8d0e9014edbe7032758e19cdb3aa6b470d11815b`

The collector additionally hashes this preregistration, the exact VG006 and
SF016 reports, fresh float32 extension, Git commit, and runtime before the
first action.

## Safety and next gate

Student updates, teacher actions executed, FlightSim packets, N712 accesses,
and Submission authorization remain zero. If VG007 admits, preregister one
source-balanced aggregate refit that starts from VG005 and gives clean VG003
and VG007 equal objective weight per update with 256-step recurrent windows.
No live work is authorized.
