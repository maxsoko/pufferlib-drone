# VQ2 LC144 phase-10 CEM milestone preregistration

Screen source-locked LC143 against its LC141 parent in one 256-row native
vector with two identical 128-seed groups and a 15,000-step horizon.  Each
policy executes as a complete recurrent Puffer actor in a source-matched
256-row paired batch.  LC143 must differ from LC141 only in the indexed
phase-10 residual output-bias row.

Use raw index 11 as the bounded target.  Select LC143 only if it creates at
least one paired raw-11 gain, zero paired loss, exact transport, monotonic held
progress, and no unresolved trajectory.  Otherwise reject LC143 and keep LC105
as the safe frontier.

This is offline evidence only.  It sends no FlightSim packet and grants no live
or Submission authority.  The proxy has 24 gates; direct simulator inspection
indicates approximately 20 official gates or more, and official finish status
is the only official lap proof.
