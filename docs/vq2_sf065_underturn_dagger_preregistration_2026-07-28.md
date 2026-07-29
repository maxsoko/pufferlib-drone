# VQ2-SF065 safe-under-turn DAgger — 2026-07-28

Tag: `vq2_sf065_underturn_dagger_64`

Run frozen SF063 on `64` exact measured-course native instances with seed
`42065`, zero teacher blend/randomization, and a `1600`-step cap. SF063 emits
every plant action. Query the frozen SF016 oracle only for labels at SF063's
visited states, including public phase `1/6`; never execute, mix, clip,
schedule, or arbitrate with a teacher action.

Preserve every genuine crash, missed-gate, or out-of-order terminal as a final
legal DAgger row. Require 64 native terminal episodes, no timeout,
action/rate/thrust fault, malformed episode boundary, phase decrease/off-tick
change, label-count error, or delivered-action error above `1e-7`. Record phase
counts and terminal categories.

A pass permits only an offline aggregate fit retaining both SF062's prior
post-Gate-1 crash states and SF065's new safe under-turn states, plus SF049's
clean anchor. It is not a completion claim. Send zero FlightSim packets, never
access N712, and do not authorize a screen, shadow, bounded flight, or
Submission.
