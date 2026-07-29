# VQ2 VG008 variable-gate DAgger round 1 preregistration — 2026-07-29

## Trigger and correction

VG006 rejected the sole VG005 checkpoint at `0/256` and localized the first
teacher-free failure to Gate 1 -> Gate 2. VG007 then executed one 256-episode
student rollout but rejected its dataset: naturally terminating trajectories
produced `60,887` records versus an arbitrary `100,000`-record floor. The
retained legal arrays prove 242/256 episodes reached held public phase 1 and
18/256 reached phase 2, with one final terminal per episode and 27,982 phase-1
records. VG007 did not persist native metrics on rejection, so it cannot prove
that record count was the only failed overall predicate. Its dataset remains
rejected and is never trained on or retried unchanged. Postmortem report SHA is
`9f494eaa6d763380307b9a371f8b7c2d05e10f4eeb5b871b8b42126f131d444d`.

VG008 corrects both evidence defects without weakening admission. It doubles
the independent population to 512 episodes while retaining the 100,000-record
floor, and it writes a complete JSON rejection report containing every native,
transport, layout, phase, and volume predicate before raising. It uses a new
seed and tag: `vq2_vg008_variable_gate_dagger_round1_512`.

## Fixed rollout and storage

- Seed `429048`; 512 fixed instances/episodes; exact-uniform construction-time
  gate counts 5 through 12 (64 each).
- Full starts and admitted randomized reachable geometry, true `0.75 m`
  aperture. Native termination or 2,048 policy steps (32 seconds) bounds every
  episode.
- VG005 checkpoint SHA
  `f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3`
  emits every complete four-channel plant action as its deterministic mean.
  Both teacher action blend and every teacher actuation path are zero.
- The SF016-admitted alignment-oracle query labels the VG005-visited state but
  never drives, blends, clips, overrides, or schedules an action. Oracle source
  SHA is
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`;
  parity report SHA is
  `a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`.
- Each record persists only `mask[4096] uint8 + legal tail[22] float32 + held
  public phase[1] float32 + oracle action[4] float32`, plus terminal/valid
  flags. No total count or privileged native value is stored.

The held public phase is sampled at 4 Hz and is official index divided by the
engine cap 16. Every instance must have one contiguous valid prefix and exactly
one terminal at its last valid row. The next native action-history newest slot
must reproduce the VG005 action actually executed within `1e-7`.

## Admission

Require all of the following simultaneously:

- 512 native terminal logs and exactly 1/8 episode mass at each count 5--12;
- at least 100,000 records and at least one phase-1 record;
- at least 90% Gate-1 reach and 1% Gate-2 reach;
- crash rate at most 5%, with zero out-of-order, crossing-margin, action,
  wire-rate, or thrust-envelope violation;
- finite/enveloped student and oracle actions, executed-action error at most
  `1e-7`, exact `/16` phase encoding, and zero off-tick phase change, decrease,
  or skip;
- exact terminal/prefix layout and record/episode-length agreement.

Miss and timeout outcomes are allowed because this is a failure-state label
distribution, not a policy admission. An admitted dataset is paired with the
immutable VG003 clean corpus at equal source objective weight; raw clean record
volume may never dilute DAgger states.

## Resume, rejection, and evidence

Before rollout, sibling state locks the Git commit, executable source hashes,
runtime, float32 extension, seed, dimensions, horizon, and safety fields. An
interrupted process may resume only under exact identity. Admission rejection
writes a sibling JSON report with the complete predicate map and terminal state
before raising; the tag is then permanently closed. Admission finalizes exact
used-length arrays and hashes every file.

Bootstrap must pass both native suites, the float32 ABI assertion, and focused
Python tests. Once state exists it may not install, build, or rerun tests.
State, report, metadata, checkpoint references, and runner logs are synced back
continuously.

## Frozen source surface

- `scripts/collect_vq2_variable_gate_dagger.py` — `46585ea6dfe6bb96ebf669ac7c9c2fb9ce8602b84b8e363d706790b4bd9cc8c6`
- `scripts/run_vq2_vg008_vast.sh` — `e6cb8c3e6600926b7d49346222ab8d01c214673cd359b723f7163c2e030c12e0`
- `tests/test_collect_vq2_variable_gate_dagger.py` — `b734099d2c3e1aa43279f57cec89e37fbcbbbc5e0da0c4ea053b93500633a79c`
- `pufferlib/vq2_oracle.py` — `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_public_phase.py` — `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` — `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `scripts/eval_vq2_variable_gate_recurrent_policy.py` — `e4ca39c4f66ef2348c123f9d8d0e9014edbe7032758e19cdb3aa6b470d11815b`
- VG006 rejection report and SF016 oracle-query parity report

The collector hashes every item, this preregistration, the compiled extension,
runtime, and commit before its first plant action.

## Safety and next gate

Student updates, teacher actions executed, FlightSim packets, N712 accesses,
and Submission authorization remain zero. VG008 admission authorizes only one
source-balanced recurrent refit starting from VG005, followed by a new-tagged
teacher-free screen. It authorizes no shadow, live attempt, or Submission use.
