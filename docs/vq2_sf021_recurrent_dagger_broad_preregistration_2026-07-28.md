# VQ2-SF021 broad SF014-driven DAgger preregistration — 2026-07-28

Tag: `vq2_sf021_recurrent_dagger_broad_512`

SF019 fixes SF018's catastrophic forgetting, but misses its fixed per-channel
admission because SF017 held-out roll MSE is `0.0631558`. The error is not
broad: agent `62` alone contributes roughly `0.2929` roll MSE over `221`
records, concentrated near steps `194..220`, while six of eight held-out
episodes remain below `0.008` roll MSE. Only the two long held-out episodes
represent the sparse post-Gate-1 transition regime. SF020's extra roll weight
does not improve that trajectory (`0.0658753` roll MSE).

SF021 tests the simplest diagnosis: insufficient student-state coverage. It
changes no model, loss, oracle, plant, legal ABI, or action source. It repeats
SF017 with eight times as many fresh randomized courses.

## Fixed collection

- Use unchanged SF014 checkpoint SHA-256
  `00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a`.
- Use the SF016-admitted query/report SHA-256 values
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`
  and `a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d`.
- Run `512` fresh randomized full-course episodes, one per native agent, seed
  `42021`, fixed `0.75 m` aperture and plant.
- Bound collection to `512` control steps. SF015/SF017 terminated by step
  `250`; any unfinished episode rejects collection and leaves no permanent
  dataset.
- Native teacher blend and every native teacher switch remain exactly zero.
  SF014's deterministic recurrent mean emits every complete plant action.
- Query the admitted alignment oracle only at the same current student state,
  before the step, and persist only the legal observation plus its four-channel
  label. Never persist the privileged query input.
- Require the next public action history to equal the executed SF014 action
  within `1e-7`.

Persist time-major `mask/tail/action/terminal/valid` arrays, exact episode
lengths, hashes, and zero privileged values per record.

Run once:

```bash
.venv/bin/python scripts/collect_vq2_recurrent_dagger_broad.py
```

Frozen SHA-256 values:

- broad collector:
  `090067c783f180f24be809efe6d5eb53b93eed222c67d8b5a11abf9a2f1db08d`;
- broad collector tests:
  `26dc30718428468bdb94e2a8dbe9ca9954bbc589e6324f00013f97f2204cc256`;
- frozen SF017 collector and tests:
  `808524dfdd6a0e47c643dfaf852e83618382afd850743b922a476f9ef173b7c0` /
  `7425ef2458e7c71b575844d0eae446ab46d738403458958e17848277878ecbd4`;
- admitted query and recurrent actor:
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694` /
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- legal writer and actor load path:
  `84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868` /
  `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa`.

The broad/SF017 collectors, oracle query/parity, legal writer, and actor
contract suite passes `20/20` before collection.

## Admission

Require all `512` episodes to terminate cleanly within `512` steps, zero crash,
timeout, out-of-order, action, wire-rate, and thrust violations, one terminal
at each episode's final valid record, exact label count, finite bounded labels
and actions, executed-action error at most `1e-7`, and zero stored privilege.
Missed gates are retained because they are the student-driven DAgger states.

Passing admits only a separately preregistered aggregate fit. It does not admit
SF014, SF019, a future child, a teacher blend, a native screen, or FlightSim.

## Safety boundary

Offline native collection only. Send zero FlightSim packets, perform zero
updates, do not access N712, and do not shadow, reset, arm, setpoint, run a
bounded attempt, or select Submission.
