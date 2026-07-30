# VQ2 VG009 corrected variable-gate DAgger round 1 preregistration — 2026-07-30

## Authority and causal change

The persistent goal prompt SHA-256
`052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`
explicitly supersedes the former two-failure stop and authorizes a new VG009
tag/seed after diagnosis. VG008 diagnosis SHA-256
`f9d99a436eaa300f70e52c00dc40b9f5752aad0684aac163ad7de7f439f183a4`
requires exactly two causal corrections:

1. rename the recurrent actor result so it cannot shadow the output `Path`,
   and atomically persist the complete predicate map in both rejection report
   and terminal state before raising; and
2. retain `crossing_margin_violation` as a recorded quality diagnostic but do
   not reject a DAgger failure-state corpus solely for an accepted crossing in
   the `0.50--0.75 m` annulus of the true `0.75 m` aperture.

Every other VG008 collection and admission choice remains fixed. VG007 and
VG008 arrays are permanently quarantined and are not read, resumed, finalized,
or used for training. The new tag is
`vq2_vg009_variable_gate_dagger_round1_corrected_512`; seed is `429049`.

## Fixed rollout and storage

- Run exactly 512 fixed instances/episodes with construction-time gate counts
  5 through 12, exactly 64 of each, on admitted randomized reachable geometry.
- Use full randomized starts, true `0.75 m` aperture, and a 2,048-policy-step
  (32-second) bound. Native terminal ends each instance earlier when reached.
- VG005 checkpoint SHA-256
  `f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3`
  emits every complete four-channel plant action as its deterministic mean.
  Teacher action blend and every teacher actuation path remain zero.
- The admitted SF016 alignment oracle only queries labels at VG005-visited
  states. Oracle source SHA-256 is
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`;
  parity report SHA-256 is
  `a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`.
- Persist only `mask[4096] uint8 + legal tail[22] float32 + held public
  phase[1] float32 + oracle action[4] float32`, plus terminal and valid flags.
  Persist no native state, gate count, course geometry, or other privilege.

The public phase is sampled at 4 Hz and encodes clamped official progress
divided by the engine cap 16. Each instance must have one contiguous valid
prefix and exactly one terminal on its final valid record. The next native
action-history newest slot must reproduce every executed VG005 action within
`1e-7`.

## Admission and diagnostics

Admission requires all of the following simultaneously:

- 512 native terminal logs and exact `1/8` episode mass at each gate count
  5--12;
- at least 100,000 causal records, at least one phase-1 record, at least 90%
  Gate-1 reach, and at least 1% Gate-2 reach;
- crash rate at most 5%, with zero out-of-order, normalized-action-envelope,
  wire-rate-envelope, and thrust-envelope violation;
- finite/enveloped student and query actions, terminal-inclusive executed-
  action error at most `1e-7`, exact `/16` phase encoding, and zero off-tick
  phase change, decrease, or skip; and
- exact terminal/prefix shapes and record/episode-length agreement.

Miss and timeout outcomes remain permitted because this is a student-visited
failure-state label distribution, not policy admission. The native
`crossing_margin_violation` metric is included verbatim in admitted and
rejected reports with `admission_predicate=false`, diagnostic radius `0.50 m`,
and official aperture radius `0.75 m`. Crash and every hard transport/safety
predicate above remain admission gates.

Before any rejection exception, VG009 writes a sibling rejection report with
the complete named Boolean predicate map, then atomically writes terminal
state containing the same map, the failed predicate list, and report hash.
Unit tests exercise both the crossing-margin demotion and this persistence
path. A rejected tag is immutable and cannot resume.

## Resume and remote execution

The sibling state is written before the first plant action and locks Git
commit, source hashes, float32 compiled extension, runtime, tag, seed,
population, horizon, and safety fields. A genuine process interruption with
status `collecting` may resume only under exact identity. Once state exists,
the Vast wrapper may not install, rebuild, or rerun tests. An admitted run
finalizes used-length arrays, hashes each file, writes report/metadata/state,
and removes only its uniquely tagged staging directory.

Bootstrap must pass both native regression suites, float32 ABI assertion, and
the focused/adjacent Python suites before the only rollout. The wrapper
requires at least 32 visible CPUs and 15 GiB free storage.

## Frozen source surface

- `scripts/collect_vq2_variable_gate_dagger.py` — `4f52642a29b5dc9257615f7e54c4169b038dfad70ba8b7a5034d0e91d521e12a`
- `scripts/run_vq2_vg009_vast.sh` — `5ad42e7ec686c70120a2290064b556a45cba95ae06d55143a22e15f430d25550`
- `tests/test_collect_vq2_variable_gate_dagger.py` — `a3dbba9a1e37290980503f2dac10b2feb1a2345aabd5917d4af1898bed916452`
- persistent goal prompt — `052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`
- VG008 diagnosis — `f9d99a436eaa300f70e52c00dc40b9f5752aad0684aac163ad7de7f439f183a4`
- `pufferlib/vq2_oracle.py` — `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
- `pufferlib/vq2_informed.py`, `pufferlib/vq2_public_phase.py`,
  `pufferlib/vq2_recurrent_phase.py`, the VG003 writer, VG006 evaluator and
  report, and SF016 parity report are hashed into state before rollout.

The collector records the actual preregistration, compiled extension, runtime,
and source commit hashes before the first action. The pushed commit is supplied
to the wrapper through `VQ2_EXPECTED_COMMIT` and must match remote `HEAD`.

## Safety and next gate

This is offline native data collection only. Student updates, teacher actions
executed, FlightSim packets, N712 accesses, and Submission authorization are
all zero. VG009 admission authorizes only a newly preregistered source-balanced
recurrent refit starting from VG005 with VG003 clean anchors and VG009 DAgger
records at equal source objective weight, followed by a new teacher-free
screen. It authorizes no Windows shadow, bounded Training run, or Submission.
