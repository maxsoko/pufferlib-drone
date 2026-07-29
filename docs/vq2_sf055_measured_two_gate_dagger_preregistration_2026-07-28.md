# VQ2-SF055 second exact measured-course DAgger — 2026-07-28

Tag: `vq2_sf055_measured_two_gate_dagger_64`

Run frozen SF053 on `64` exact measured native instances with seed `42055`,
zero teacher blend/randomization, the `0.75 m` aperture, and a `768`-step cap.
SF053 emits every plant action; the admitted SF016 alignment oracle is queried
only for labels on the visited states. Preserve genuine terminal boundaries
and store only legal observations, the one 4 Hz held public phase scalar,
oracle labels, valid bits, and terminal bits.

Require `64` real final terminals, no timeout/out-of-order/action/rate/thrust
fault, monotonic tick-only phase, exact label count, and delivered student
action error at most `1e-7`. The expected Gate-1 miss is training data, not an
admission success. A data pass permits only a three-source offline fit anchored
by SF049 and SF052. Send zero FlightSim packets, never access N712, and do not
authorize a screen, shadow, bounded flight, or Submission.
