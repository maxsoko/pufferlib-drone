# VQ2 LC126 phase-6 on-policy coordinate ES preregistration

LC125 proves that LC123's new phase-8 distribution is recoverable to raw 10,
but only three trajectories enter the intervention window and its `1` success /
`2` failure corpus fails the preregistered whole-trajectory split. Do not fit
LC125. Retain LC105 and pivot away from teacher-action regression.

Run one closed-loop coordinate-ES generation around LC105's phase-6 indexed
Puffer output bias. Use eight 32-seed groups at seed `432050`: LC105, then
antithetic pitch `+-0.0025`, roll `+-0.0125`, and thrust `+-0.00625`
pre-tanh perturbations. The eighth candidate adds the half-amplitude joint
direction `[+0.0025,-0.0125,+0.00625,0]`. All groups execute through one
complete 256-row LC105 recurrent actor; exact per-group output-bias surgery is
applied only at held public phase 6. This preserves the established actor batch
while testing eight plant trajectories in one native vector.

Use the fixed 24-gate proxy, 12,000 steps, and raw index 10 as the causal
milestone. Select a nonzero candidate only if it creates at least one paired
raw-10 gain, zero loss, no added pre-target terminal, and exact finite action,
transport, and public-progress contracts. A selection authorizes one 128-pair
confirmation only. If no candidate reaches raw 10, use the source-locked raw
index distributions only to construct one diagonal on-policy proposal; do not
promote from lower progress.

This is offline Puffer optimization, not an official lap. Direct simulator
inspection shows approximately 20 official gates or more, and only a
nonnegative official finish time proves completion. Send zero FlightSim
packets and keep VQ2 Submission forbidden.
