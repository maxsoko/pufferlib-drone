# VQ2 LC128 phase-6 on-policy PPO milestone preregistration

LC127 completes four teacher-free stochastic rollouts and three finite
phase-6-only PPO updates. Its selected rollout-4 state improves stochastic mean
maximum raw progress from `3.3671875` to `3.375`, while all transport and frozen
state predicates pass. No stochastic rollout reaches raw 9 or 10; training is
not promotion evidence.

Run exactly one deterministic parent-versus-candidate screen with 128
duplicated seeds per group at seed `432050`, the fixed 24-gate proxy, a
12,000-step horizon, and raw index 10 as the milestone. Execute LC105 and the
selected LC127 state as two complete saved-form Puffer actors, each over a full
256-row baseline-plus-candidate context. Exploration is exactly zero and no
teacher is queried or executed.

Select LC127 only if it creates at least one paired raw-10 gain, zero loss, no
added pre-target terminal, and exact finite action, transport, public-progress,
and source-lock contracts. Selection authorizes one independent offline
confirmation only. Otherwise reject LC127 and retain LC105.

This is an offline proxy milestone. The official VQ2 course has approximately
20 gates or more by direct simulator inspection; only a nonnegative official
finish time proves a lap. Send zero FlightSim packets and keep Submission
forbidden.
