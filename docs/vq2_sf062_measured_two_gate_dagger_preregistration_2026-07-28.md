# VQ2-SF062 current-distribution DAgger — 2026-07-28

Tag: `vq2_sf062_measured_two_gate_dagger_phase1_second_64`

Run frozen SF059 on `64` exact measured-course native instances with seed
`42062`, zero teacher blend/randomization, and a `1600`-step cap. SF059 emits
every plant action. Query the frozen SF016 oracle only for labels at states
SF059 visits, including public phase `1/6`; never execute, mix, clip, schedule,
or arbitrate with a teacher action.

Preserve each genuine crash, missed-gate, or out-of-order terminal as a final
legal DAgger row. Require 64 native terminal episodes, no timeout,
action/rate/thrust fault, malformed episode boundary, phase decrease/off-tick
change, label-count error, or delivered-action error above `1e-7`. Record phase
counts and terminal categories.

This is an offline data collection, not a completion claim. A passing
collection permits only a separately preregistered fit that retains clean and
prior DAgger anchors while giving the current SF059 distribution an explicit
source-specific validation gate. Send zero FlightSim packets, never access
N712, and do not authorize a screen, shadow, bounded flight, or Submission.
