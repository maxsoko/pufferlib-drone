# VQ2 LC149 phase-11 state-dependent milestone preregistration

Screen source-locked LC148 against its LC143 parent in one 256-row native
vector with two identical 128-seed groups and a 17,000-step horizon. Each
policy executes as a complete recurrent Puffer actor in a source-matched
256-row paired batch. LC148 must differ from LC143 only in all four indexed
phase-11 residual parameter rows; every other saved tensor and every other
phase row must remain exact.

Use raw index 12 as the bounded target. Select LC148 only if it creates at
least one paired raw-12 gain, zero paired raw-12 losses, exact transport,
monotonic held progress, and no unresolved trajectory. Otherwise reject LC148
and retain LC105 as the safe frontier.

This is teacher-free offline evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
