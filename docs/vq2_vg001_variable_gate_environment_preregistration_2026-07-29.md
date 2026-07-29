# VQ2-VG001 variable-gate environment preregistration — 2026-07-29

Tag: `vq2_vg001_variable_gate_environment_contract`

The authoritative variable-gate prompt supersedes the fixed-six-gate
assumption. This cell changes native/offline infrastructure only. It sends no
FlightSim packet, reads no consumed sealed test, trains no student, and cannot
authorize VQ2 Submission.

## Fixed implementation contract

- Add default-off native binding keys that select one fixed gate count per
  vector environment. With randomization enabled, assign counts uniformly in
  the inclusive configured range. VG001 fixes the range to `5..12`; a vector
  of 512 one-drone environments must contain exactly 64 instances of every
  count, including the six-gate regression anchor. Gate count cannot change on
  episode reset.
- Preserve every historical environment when the new switch is disabled.
  Gate count remains the existing `num_gates` kwarg and the historical RNG
  stream must not be consumed or shifted.
- Extend the existing SF010/SF011 per-agent position randomizer with a
  default-off bounded rejection check. VG001 requires strictly increasing gate
  planes, no intersecting apertures, and a configured maximum adjacent-center
  distance. Retain the existing geometry-scale and independent position-jitter
  draws; do not introduce a new course generator.
- Extend ordered-crossing logging through the existing engine cap of 16 gates
  and add per-count episode/success rates. These are diagnostics only and may
  not affect observation, action, reward, or termination.
- Preregister the sole public phase encoding as
  `clamp(active_gate_index, 0, 16) / 16`. Centralize the denominator and held
  4 Hz update cadence in one module for all new variable-gate collection,
  screening, export, and deployment code. Historical `/6` artifacts remain
  frozen evidence and are not silently reinterpreted.

## Regression and acceptance checks

1. Native unit tests prove fixed-count selection, exact uniform `5..12`
   assignment over 512 instance indices, seed rotation without count bias,
   and unchanged fixed-six behavior when disabled.
2. Course tests exercise every count `5..12` over deterministic randomized
   seeds and require ordered, finite, non-overlapping, bounded-length geometry.
3. Observation tests require the native legal visual ABI to retain width
   `4118`; identical active-gate/state fixtures at different total counts must
   have byte-identical `4096` masks and `22`-value legal tails. Appended public
   phase is exactly index divided by 16 and contains no total-count value.
4. Existing focused oracle, visual-native, phase-actor, and collector contract
   tests remain green.

VG001 admits only the environment/ABI implementation. The separately tagged
oracle screen must still finish at least 99% of 512 randomized courses for each
count in `{5, 8, 11, 12}` at radius `0.75 m`, with zero collision, before a
variable-count legal corpus can be collected.

## Safety boundary

Offline native code and tests only. FlightSim remains frozen after N522. Send
zero heartbeat, TIMESYNC, metadata, reset, arm/disarm, or setpoint packets.
Never select VQ2 Submission and never access N712's consumed test.
