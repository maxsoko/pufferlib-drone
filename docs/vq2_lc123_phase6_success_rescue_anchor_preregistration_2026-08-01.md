# VQ2 LC123 phase-6 success-rescue anchor preregistration

LC117 proves that the training-only alignment oracle can first rescue LC105 to
raw index 10 when it takes control at held public phase 6. LC119 captures the
exact 60,082 intervention records: four queried trajectories reach raw 10 and
fifteen still fail. LC120 fits the oracle on both outcome classes, but LC121
and LC122 show that its phase-6 endpoint, alone or composed with phase 8,
creates no raw-10 completion. Do not retry that objective.

Fit one whole-Puffer phase-6 residual endpoint from LC119. On the four rescued
trajectories, target the training-only oracle action. On the fifteen failed
trajectories, target LC105's frozen action so the fit does not imitate an
intervention that failed. Split whole trajectories within each outcome class,
sample outcome- and trajectory-balanced records, and train only the four
phase-6 residual tensors for 512 Adam steps. The encoder, recurrent base, base
action head, ABI, and all other public-phase heads remain frozen.

At each validation interval, evaluate parameter scales `0.1,0.3,0.5,1.0`.
Select the finite nonzero candidate with the lowest held rescued-trajectory
teacher MSE while holding failed-trajectory drift from LC105 to at most
`0.00025`. Admit the numerical fit only if held rescued-trajectory teacher MSE
improves by at least `1.20x`, phase-6 parameter delta L2 is at most `64`, and
the failure drift bound holds. A passed fit authorizes exactly one reduced
teacher-free parent-versus-candidate raw-index-10 screen; it does not promote a
checkpoint or authorize FlightSim.

The offline course is a fixed 24-gate proxy. The official VQ2 course remains
approximately 20 gates or more by direct simulator inspection, and only a
nonnegative official finish time proves a completed lap. This job sends zero
FlightSim packets and VQ2 Submission remains forbidden.
