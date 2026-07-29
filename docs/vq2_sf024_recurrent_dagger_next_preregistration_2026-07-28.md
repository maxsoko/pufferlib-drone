# VQ2-SF024 SF022-driven DAgger iteration — 2026-07-28

Tag: `vq2_sf024_recurrent_dagger_next_512`

SF023 is rejected: its source-locked teacher-free report SHA-256 is
`967dac29339550cfc30f8bda7b31d477ce5710d04eeb30c2a316035b74627b07`.
It completes zero courses, passes mean `0.220703` gates, and terminates with
native crash in `0.921875` of episodes, almost entirely low-floor events. This
is closed-loop distribution shift after low supervised validation error.

SF024 performs the next standard DAgger query on those missing states. A native
training crash is a terminal label source, never an authorized runtime safety
exception.

## Fixed collection

- Use unchanged SF022 checkpoint/report SHA-256
  `57949ba391ffdc4d1e0dda5aa352fbb7b6dcb2a51b1f7742018926f6c6b2d903` /
  `2cfc253ef13dc4a438a784df595a52947aaa0bcdb42320a6069bebdfa85f2f66`.
- Run `512` new randomized courses, one per agent, seed `42024`, fixed `0.75 m`
  aperture/plant, with a collection cap of `2048` control steps.
- SF022's deterministic recurrent mean emits every complete plant action.
  Teacher blend and all native teacher switches remain zero.
- Query the SF016-admitted oracle only before each current student-driven step.
  Persist only the legal observation and complete four-channel query label;
  persist no privileged query input.
- End each episode at its first native terminal, including crash. Require every
  episode to terminate within the cap and retain the pre-terminal causal record.
- Verify public action history equals SF022's emitted action within `1e-7`.

Run once:

```bash
.venv/bin/python scripts/collect_vq2_recurrent_dagger_next.py
```

Frozen SHA-256 values:

- SF024 collector/tests:
  `83e86b22e5ff6faa53118998b5ca7603a6520e9de37b94ab7925a16ee3a8ebaa` /
  `e7c07db949fbef5c21dca152a46d4846c460e6e670b5e72e3eee28a752c8d419`;
- SF021 collector/tests:
  `090067c783f180f24be809efe6d5eb53b93eed222c67d8b5a11abf9a2f1db08d` /
  `26dc30718428468bdb94e2a8dbe9ca9954bbc589e6324f00013f97f2204cc256`;
- SF023 evaluator, admitted query, recurrent actor, and legal writer:
  `0e99ac443cc2d9c1dde8a1d4efe3efe3e180cd72fb0358b630de5615e2869b2d` /
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694` /
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199` /
  `84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868`.

The next/broad DAgger collectors, SF023 evaluator, oracle query, and actor suite
passes `12/12` before collection.

## Admission

Require all `512` episodes and exact final-terminal layouts, zero timeout,
out-of-order, action, wire-rate, thrust, non-finite, or executed-action fault,
label count equal to the valid-record sum, and zero stored privilege. Native
crash and missed-gate terminals are explicitly retained as DAgger states and
reported; they do not admit the actor or weaken any later zero-crash screen.

A pass admits only a separately preregistered aggregate retrain. Never retry
SF023 or SF024 unchanged.

## Safety boundary

Offline native training-data collection only. Send zero FlightSim packets, do
not access N712, and do not shadow, reset, arm, setpoint, run a bounded attempt,
or select Submission.
