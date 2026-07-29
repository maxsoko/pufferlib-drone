# VQ2-SF017 SF014-driven DAgger dataset preregistration — 2026-07-28

Tag: `vq2_sf017_recurrent_dagger_dataset_64`

SF015 rejects SF014 for direct promotion after `21/64` Gate-1 passes and zero
Gate-2 passes. SF016 admits a training-only query of the simple alignment
governor: over `403722` transitions, maximum pitch/roll/thrust/yaw differences
from the native executed oracle are
`5.58e-6/2.83e-6/2.83e-6/0`, below the fixed `5e-5` threshold.

SF017 collects the first actual DAgger distribution. The recurrent student is
the only plant action source; the admitted oracle query labels the same current
states without acting.

## Fixed collection

- SF014 checkpoint SHA-256:
  `00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a`.
- SF016 report SHA-256:
  `a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`.
- Run `64` disjoint randomized full-course episodes, one per agent, seed
  `42017`, fixed `0.75 m` aperture and plant, and `180 s` horizon.
- Set native teacher blend and all teacher controller switches to zero. Use the
  deterministic SF014 joint four-channel mean for every active plant step.
- Advance SF014's recurrent state causally from the genuine full start and
  freeze it only after that agent terminates.
- Before each student-driven step, query the SF016-admitted oracle from the
  current privileged state. Persist the current legal observation and that
  complete four-channel label; never persist the privileged query input.
- Verify after the step that the next native executed-action history equals
  SF014's emitted action within `1e-7`.

Persist the same time-major legal-only format as SF012:

- `mask.npy [time,64,4096] uint8`;
- `tail.npy [time,64,22] float32`;
- `action.npy [time,64,4] float32` containing query labels;
- `terminal.npy` and `valid.npy`; and
- exact metadata, file hashes, episode lengths, source hashes, and report.

Run once:

```bash
.venv/bin/python scripts/collect_vq2_recurrent_dagger.py
```

Frozen SHA-256 values:

- collector: `808524dfdd6a0e47c643dfaf852e83618382afd850743b922a476f9ef173b7c0`;
- collector tests:
  `7425ef2458e7c71b575844d0eae446ab46d738403458958e17848277878ecbd4`;
- oracle query: `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`;
- legal writer: `84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868`;
- teacher-free evaluator/load path:
  `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa`.

The DAgger collector, query, parity, legal writer, and actor contract suite
passes `18/18` before collection.

## Admission

Missed gates are expected and may be retained because they define the DAgger
state distribution. Require all `64` episodes to terminate cleanly, zero
collision, timeout, out-of-order, action, wire-rate, and thrust violation,
finite bounded student actions and labels, one final terminal per contiguous
episode, label count equal to summed lengths, executed student action error at
most `1e-7`, and zero persisted privileged values.

A pass admits this dataset only for a separately preregistered aggregate
SF012+SF017 recurrent fit. It does not admit SF014, a retrained child, native
teacher blend, or FlightSim.

## Safety boundary

Native offline DAgger collection only. The student emits every executed plant
action; the teacher emits none. Perform zero updates and write no checkpoint.
Send zero FlightSim packets, do not touch N712, and do not shadow, reset, arm,
setpoint, run a bounded attempt, or select Submission.
