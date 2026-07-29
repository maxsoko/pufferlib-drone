# VQ2-SF012 legal-observation oracle dataset preregistration — 2026-07-28

Tag: `vq2_sf012_oracle_legal_bc_dataset_64`

SF011 admits one simple training-only oracle after `4096/4096` randomized
six-gate finishes with zero safety or action-envelope violation. SF012 may run
that unchanged oracle exactly once to produce the first student-training
dataset. It does not train, screen, or admit a student.

## Frozen collection

Run:

```bash
.venv/bin/python scripts/collect_vq2_oracle_bc_dataset.py
```

Use exactly `64` vector agents, one complete episode per agent, seed `42012`,
the SF011 randomized-course distribution, fixed `0.75 m` gate radius, fixed
plant, full course starts, `2.0 m/s` governed target, and `180 s` horizon. The
policy-action buffer is all zero and `teacher_action_blend=1`; the admitted
alignment governor therefore emits the complete executed CTBR action.

Frozen SHA-256 values before collection are:

- native controller C: `47f5c40d0cd04e59caf0e3aad604847a8273deb7c4e64f6f683c2a813b2ab5b4`;
- native header: `b202dbf4ce4a142da16abe044f84e63e76313cf04d351147b757525608ef2df8`;
- binding: `e30b734ee467ac8f5ce2d79f00df9a8d2738b871097317126c6a3c7f242729e2`;
- oracle evaluator: `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`;
- VQ2 INI: `5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b`;
- legal ABI module: `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`;
- compiled CPU extension: `2a61ff24489e01c85b678317684ae8fe8607ac6960ee46f5efa48b5ff265ea62`;
- collector: `84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868`;
- focused tests: `efcd566c0302f4eaf01b8b98810173f068fd1b415e9e3d3bee42151875e1f0af`;
- SF011 report: `f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e`.

The focused legal-storage, action-timing, full-history-layout, source-contract,
and oracle evaluator suite passes `25/25` before collection.

## Dataset boundary

Write time-major arrays truncated to the longest used episode:

- `mask.npy`: `[time, 64, 4096]`, `uint8`, decoded by division by `255`;
- `tail.npy`: `[time, 64, 22]`, `float32`;
- `action.npy`: `[time, 64, 4]`, `float32`;
- `terminal.npy`: `[time, 64]`, `uint8`;
- `valid.npy`: `[time, 64]`, `uint8`; and
- `metadata.json` with exact episode lengths, shapes, dtypes, file hashes,
  source hashes, and ABI schemas.

For each active transition, save legal observation `O_t`. After the native
step, recover its exact executed action from the newest four-value action slot
of `O_(t+1)`. Require the remaining eight history values in `O_(t+1)` to equal
the prior eight newest values in `O_t` within `1e-7`. This includes the action
that produces a terminal transition.

The native observation width is `4152`, but persistent actor data must contain
only the leading `4118` values split as `4096 + 22`. Never save, serialize, or
expose the trailing `34` privileged values to the student actor. Mask
quantization has a fixed maximum reconstruction error of `0.5/255`; all legal
sensor and action values remain `float32`.

## Admission

Materialize the permanent dataset only if all of these hold:

- `64/64` valid ordered six-gate finishes;
- zero collision, miss, out-of-order, timeout, crossing-margin, action,
  wire-rate, or thrust-envelope violation;
- every ordered gate sampled rate is exactly one and every mean radial crossing
  error is at most `0.10 m`;
- every episode has a nonempty contiguous valid prefix, exactly one terminal,
  and that terminal is its final saved transition;
- saved label count equals the sum of episode lengths;
- every label is finite and in `[-1, 1]`; and
- maximum audited action-history shift error is at most `1e-7`.

Failure leaves no permanent dataset under this tag. A pass admits only this
dataset for offline recurrent BC/DAgger. It does not admit any student
checkpoint, native policy screen, FlightSim shadow, reset, arm, setpoint,
bounded attempt, or Submission action.

## Safety boundary

Native offline collection only. Send zero FlightSim packets, perform zero
student updates, write zero student checkpoints, and do not access the consumed
N712 sealed test. VQ2 Submission remains forbidden.
