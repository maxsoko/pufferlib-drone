# VQ2 VG057 warmed phase-residual fit — 2026-07-31

Close the rejected VG053/VG055 interpolation line. Load frozen VG033 into the
existing recurrent `VQ2PhaseResidualActor`, preserving its encoder, GRU,
public-phase embedding, base action head, and log standard deviation exactly.
Initialize the sole new `phase_action_residual.weight` to zero and train only
those 1,024 Puffer parameters.

Use VG039's source-locked full-start, VG033-visited causal sequences so the
residual sees deployment-shaped warmed recurrent states. Run six epochs,
256-step BPTT, eight agents per batch, AdamW `1e-3`, seed `429183`, and reserve
64 agents for validation. Public phase row weights are
`0/0.25/0.5/2/32/32...`; phase 4 is the direct Gate-4-to-5 bottleneck.

Numerical admission requires exact base parameters, at least 1,000 phase-4
validation rows, strict phase-4 improvement, at most two-percent phase-2/3
regression, and residual L2 at most `8`. Admission authorizes only a separately
preregistered teacher-free six-gate residual-scale bracket.

The first source completed all six epochs and 5,330 updates, then stopped
before checkpoint/report creation because an actor-output local shadowed the
output path. The repaired source preserves that failure as evidence, renames
the local, and selects the best numerically eligible epoch rather than the
lowest phase-4 error irrespective of phase-3 stability. It reruns from scratch
with the same deterministic training contract; the failed state is not reused.

Runtime observations remain the unchanged legal ABI, and the new learned head
emits the complete four-action correction inside one recurrent Puffer policy.
Teacher plant actions, FlightSim, shadow, Training, and Submission are
forbidden. Submission requires explicit user authorization.

Source lock before the run:

- VG033 checkpoint/VG039 report/VG056 rejection: `56a8e3b8...`/`a0bdcd0d...`/`c97082e4...`
- failed state/log evidence: `e9e90792...`/`4ef9ff5b...`; failure record: `627bca44...`
- repaired trainer/runner/test: `4da484f7...`/`f5d7014b...`/`040b7703...`
