# VQ2 LC211 LC209 phase-16 CEM preregistration

LC210 rejects the second supervised DAgger fit even though LC209's held-out
teacher-action MSE is `0.000132514`. The phase-15 lineage required a final
policy-only phase residual search after its second DAgger fit. LC211 applies
that same bounded mechanism to LC209.

Run three CEM generations of 512 exact seed-15 copies. Each trajectory is a
complete recurrent LC209 Puffer. CEM may add only one constant four-value
pre-tanh residual while public progress is phase 16; all recurrent weights,
the phase-15 adapter, the continuation adapter, and every other phase remain
frozen. The teacher/oracle is absent. Score official proxy progress first,
then native phase return. Admit only an actual raw-18 completion with exact
transport and state-surgery checks.

This is offline training only and authorizes no FlightSim or Submission run.
