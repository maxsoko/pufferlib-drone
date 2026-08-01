# VQ2 LC133 phase-8 anisotropic PPO preregistration

LC131 and LC132 prove a fast, finite, teacher-free Puffer PPO cycle, but ten
total low-noise updates produce no raw-9 pass. Reject their checkpoints. The
source-locked LC125 oracle rescue now supplies scale evidence without being
fitted: on its successful phase-8 trajectory, LC123-to-oracle action RMSE is
`0.04401613728668076`; pitch, roll, thrust, and yaw RMS gaps are approximately
`0.05293`, `0.05106`, `0.04837`, and `0.00087`. The old isotropic `0.01`
exploration is outside the useful correction scale on three channels.

Return to nondeployable LC123. Run four 512-environment stochastic rollouts at
seed `432050` with native seed-group size 128, yielding four copies of the
productive cohort. Preserve actor-batch numerics through two independent,
complete 256-row Puffer actors. Both recurrent states advance normally, and
each actor emits every channel of its complete action vectors.

Outside held public phase 8, execute the deterministic Puffer mean. At phase 8
only, sample pre-tanh actions with per-channel standard deviations
`[0.05, 0.05, 0.05, 0.002]`. Stop at raw index 9 or a native terminal. Use
only the training-native distance-progress reward at weight `5.0` plus
centered ordered-gate credit at `30.0`; set other reward terms and all teacher
terms to zero. No oracle/teacher action may be queried, mixed, or executed.

Normalize phase return across entrants and add it at weight `1.0` to raw
progress plus four-point pass credit. After each of the first three rollouts,
perform two PPO epochs at learning rate `1e-5`, clip coefficient `0.10`,
maximum gradient norm `0.5`, and `1e-4` anchor. Train only the four phase-8
indexed residual tensors; freeze the remaining Puffer state byte-exact.

Select at most one state for a deterministic raw-9 screen only if its
teacher-free gate-progress rank strictly beats rollout 1. Require finite,
nonzero return variance, parameter movement, transport, and action-envelope
checks. The LC125 labels remain quarantined and unfitted. This is a randomized
24-gate proxy for an approximately 20-plus-gate official course; only official
nonnegative finish time proves a lap. Send zero FlightSim packets and keep
Submission forbidden.
