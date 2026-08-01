# VQ2 LC131 phase-8 dense-return PPO preregistration

LC130 completed three transport-clean repeated-frontier rollouts in 282.34
seconds, but every phase-8 entrant received the same integer progress score.
Both PPO updates were exactly zero and no checkpoint was selected. Do not
repeat sparse whole-gate PPO at this frontier.

Starting from nondeployable LC123, run four 256-environment stochastic
rollouts at seed `432050` with native seed-group size 128. This creates two
copies of the only productive seed cohort while preserving one complete
256-row Puffer actor execution. Outside held public phase 8, execute the
deterministic Puffer mean. At phase 8 only, sample pre-tanh actions with fixed
standard deviation `0.01`; stop at raw index 9 or a native terminal.

Accumulate the native reward only on steps whose action was selected at held
phase 8. This is training-only reward infrastructure: the configured native
proxy supplies distance progress, centered ordered-gate credit, and body-rate
cost, while the actor still receives only its unchanged legal observation,
previous actions, recurrent state, and public progress. No teacher action may
be queried, mixed, or executed. Normalize phase-8 return across entrants and
add it at weight `1.0` to the established raw-progress plus four-point pass
score.

After each of the first three rollouts, perform four PPO epochs over the exact
rollout. Train only the four phase-8 indexed residual tensors at learning rate
`5e-5`, clip coefficient `0.10`, maximum gradient norm `0.5`, and `1e-4`
pre-update anchor. Freeze LC123 phase 6, every other phase, encoder, recurrent
base, action head, log standard deviation, and ABI byte-exact.

Select at most one state for a deterministic raw-9 screen only if its
teacher-free stochastic gate-progress rank strictly beats rollout 1. Require
finite nonzero phase-return variance and a nonzero finite parameter update;
otherwise reject the branch. This job uses a randomized 24-gate proxy. Direct
simulator inspection shows approximately 20 official gates or more, and only
`race_finish_time_ns >= 0` proves an official lap. Send zero FlightSim packets
and keep Submission forbidden.
