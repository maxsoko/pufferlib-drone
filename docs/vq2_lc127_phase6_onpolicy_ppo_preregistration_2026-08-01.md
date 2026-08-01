# VQ2 LC127 phase-6 on-policy PPO preregistration

LC126 evaluates the local phase-6 constant-bias coordinates around LC105 in
one 256-row actor context. LC105 retains one raw-9 trajectory; every nonzero
coordinate loses it, and none improves mean raw progress. Reject another
constant or diagonal bias retry. The next update must be state-dependent and
trained on closed-loop outcomes rather than oracle-action error.

Starting from frozen LC105, run four reproducible 256-agent on-policy rollouts
at seed `432050`, 24 proxy gates, and 12,000 steps. Outside held public phase 6,
execute the complete Puffer mean. At phase 6 only, sample the Puffer pre-tanh
mean with fixed standard deviation `0.01`; no teacher action is queried or
executed. Record legal recurrent hidden states, frozen base pre-tanh values,
sampled pre-tanh actions, old log probabilities, and agent identities.

After each of the first three rollouts, assign a whole-trajectory advantage
from maximum raw progress, with a four-point bonus for raw-10 completion.
Normalize across phase-6 entrants and perform four PPO epochs over the exact
rollout. Train only the four phase-6 indexed residual tensors with learning
rate `5e-5`, clip coefficient `0.10`, gradient norm `0.5`, and a `1e-4`
anchor to the pre-update tensors. Keep the camera encoder, recurrent base,
action head, ABI, log standard deviation, and every other phase byte-frozen.

Save the complete Puffer state used by each rollout. Select at most one
post-update state for a teacher-free deterministic parent-versus-candidate
screen, ranking first by raw-10 completions, then raw-9-or-later reach, then
mean maximum raw index. The selected stochastic rollout must strictly beat the
initial rollout on that rank, and every rollout/update must be finite,
in-envelope, transport-exact, and teacher-free. Training does not itself
promote a checkpoint.

This is offline Puffer optimization on a 24-gate proxy. Direct simulator
inspection shows approximately 20 official gates or more; only
`race_finish_time_ns >= 0` proves an official lap. Send zero FlightSim packets
and keep VQ2 Submission forbidden.
