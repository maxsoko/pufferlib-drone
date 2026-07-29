# VQ2-SF052 exact measured-course DAgger — 2026-07-28

Tag: `vq2_sf052_measured_two_gate_dagger_64`

Run frozen SF050 on `64` exact deterministic native instances using seed
`42052`, the measured Gate-1/Gate-2 geometry, `0.75 m` apertures, and zero
start/gate/camera/plant randomization. Bound collection at `768` steps; SF051
predicts a genuine terminal near step 486. SF050 emits every plant action.
Query the admitted SF016 alignment oracle only for labels at the states SF050
actually visits. Never blend or execute an oracle action.

Store only the 4,118 legal inputs, one `4 Hz` held normalized public gate index,
the oracle label, valid bit, and genuine terminal bit. Preserve time and episode
boundaries. Require exactly one genuine final terminal per agent, `64` native
terminal episodes, no timeout/out-of-order/action/rate/thrust fault, monotonic
tick-only public phase, exact label count, and delivered student-action error
at most `1e-7`.

The deterministic copies test and source-lock the collection path; they are not
claimed as independent robustness samples. A pass permits only an offline
aggregate fit retaining the complete SF049 oracle prefix as an anchor. Send
zero FlightSim packets, never access N712, and do not authorize a screen,
shadow, bounded flight, or Submission.
