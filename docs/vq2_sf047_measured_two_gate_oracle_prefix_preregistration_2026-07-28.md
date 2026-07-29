# VQ2-SF047 measured two-gate oracle prefix — 2026-07-28

Tag: `vq2_sf047_measured_two_gate_oracle_prefix_64`

Return to the solve-first curriculum. Collect 64 bounded 768-step native
oracle prefixes on the source-locked measured VQ2 Gate-1/Gate-2 geometry:
start `[0,0,0]`, Gate 1 `[10.78,0.084,0.56]`, and Gate 2
`[25.13,8.96,1.65]`, all in the existing native NED convention. Keep six
gates so the held public observation remains exactly `current_gate / 6`.

Use the admitted SF016 alignment oracle at zero student execution, exact
`0.75 m` apertures, zero course/start/camera/plant randomization, zero reset
position noise, seed `42047`, and the measured 64 Hz policy / 4 Hz held-status
cadence. Store 4,119 legal actor values and the complete four-channel oracle
action. Require every agent to reach held phase `>=2/6` within the prefix,
zero collision/miss/out-of-order/action/rate/thrust fault, exact executed-label
parity, monotonic tick-only phase, and zero stored training-only privilege.

This is exact-course curriculum data, not a deployment result. It sends zero
FlightSim packets, does not access N712, and cannot authorize a screen, shadow,
bounded attempt, or Submission.

