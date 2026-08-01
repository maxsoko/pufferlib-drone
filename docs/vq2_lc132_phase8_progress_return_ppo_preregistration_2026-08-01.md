# VQ2 LC132 phase-8 progress-return PPO preregistration

LC131 proves that the optimized on-policy cycle can learn in 183.66 seconds:
all three phase-8 updates were finite and nonzero, and stochastic mean return
improved from `-237.85` to `-168.39`. It still produced no raw-9 pass. Phase-8
records fell from 10,858 to 8,264, showing that the phase-local body-rate cost
can reward earlier terminal failure. Reject all LC131 checkpoints.

Return to nondeployable LC123. Run six 256-environment stochastic rollouts at
seed `432050`, native seed-group size 128, and one complete 256-row Puffer
actor execution. Outside held public phase 8, execute the deterministic Puffer
mean. At phase 8 only, sample pre-tanh actions with standard deviation `0.01`;
stop at raw index 9 or a native terminal.

For this training job only, set the native reward to gate-distance progress at
weight `5.0` plus centered ordered-gate credit at weight `30.0`. Set body-rate,
time, control, cross-track, camera-alignment, crossing-error, exit-velocity,
speed, altitude, teacher, finish, and invalid penalties to zero. The
nonterminal progress sum telescopes from phase-8 entry to final gate distance,
so ending earlier cannot avoid an accumulated negative cost. The Puffer actor
still receives the unchanged legal ABI; native geometry contributes only the
offline scalar reward. No teacher action is queried, mixed, or executed.

Normalize this phase return across entrants and add it at weight `1.0` to raw
progress plus the four-point pass score. After each of the first five
rollouts, perform two PPO epochs at learning rate `1e-5`, clip coefficient
`0.10`, maximum gradient norm `0.5`, and `1e-4` anchor. Train only the four
phase-8 indexed residual tensors; freeze the complete remaining Puffer state.

Select at most one state for a deterministic raw-9 screen only if its
teacher-free gate-progress rank strictly beats rollout 1. Require finite,
nonzero return variance and parameter movement throughout. This is a
randomized 24-gate proxy for the approximately 20-plus-gate official VQ2
course; only nonnegative official finish time proves a lap. Send zero
FlightSim packets and keep Submission forbidden.
