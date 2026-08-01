# VQ2 LC124 phase-6 success-rescue milestone preregistration

LC123 admits one phase-6 whole-Puffer endpoint from the exact LC119 rescue
corpus. Its held rescued-trajectory teacher-action MSE improves `1.214x`, its
failed-trajectory drift from LC105 is `0.0001998`, its parameter delta L2 is
`1.1863`, and every tensor outside the phase-6 residual is frozen.

Run exactly one teacher-free parent-versus-candidate screen. Use 128 duplicated
seeds per group at seed `432050`, the fixed 24-gate proxy, a 12,000-step
horizon, and raw index 10 as the milestone. Execute two complete saved-form
Puffer actors. Each actor must run over a full 256-row baseline-plus-candidate
batch so the established numerical context is preserved; do not vectorize
checkpoint surgery inside one actor.

Select LC123 only if it creates at least one paired raw-10 gain, zero paired
raw-10 losses, no added pre-target terminal, finite in-envelope actions, exact
executed-action transport, and exact public-progress sampling. A selection
authorizes only an independent offline confirmation. No selection rejects this
fit family and retains LC105.

This is an offline proxy milestone, not an official lap. The official VQ2
course has approximately 20 gates or more by direct simulator inspection, and
only `race_finish_time_ns >= 0` proves its finish. This job sends zero FlightSim
packets and VQ2 Submission remains forbidden.
