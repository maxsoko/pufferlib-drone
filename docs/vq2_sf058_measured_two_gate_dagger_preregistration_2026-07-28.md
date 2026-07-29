# VQ2-SF058 phase-1 exact measured-course DAgger — 2026-07-28

Tag: `vq2_sf058_measured_two_gate_dagger_64`

Run frozen SF056 on `64` exact measured-course native instances with seed
`42058`, zero teacher blend/randomization, and a `1600`-step cap. SF056 emits
every plant action. Query SF016 only for labels at visited states, including
the newly reached public phase `1/6`; never execute or blend a teacher action.

Preserve every genuine crash, missed-gate, or out-of-order terminal as a final
valid row because these are DAgger recovery states, not admission trajectories.
Require 64 native terminal episodes, no timeout, action/rate/thrust fault,
malformed episode boundary, phase decrease/off-tick change, label-count error,
or delivered-action error above `1e-7`. Record terminal categories explicitly.

A data pass permits only an offline four-source fit retaining SF049, SF052, and
SF055. It is not a completion claim. Send zero FlightSim packets, never access
N712, and do not authorize a screen, shadow, bounded flight, or Submission.
