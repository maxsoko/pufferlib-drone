# VQ2 LC016 native legal PuffeRL PPO — 2026-07-31

Train one native recurrent PuffeRL policy for 10,000,384 agent steps on a
1--24-gate randomized curriculum. Use 1,024 agents, 16 buffers, 128 native
threads, horizon 16, one full 16,384-sample minibatch per cycle, hidden size
256, one MinGRU layer, and save approximately every two million steps.

Enable the default-off `visual_policy_legal_only` boundary. The CUDA tensor
keeps its historical width, but the entire 34-value training-only tail must be
exact zero. The legal step-dt slot instead carries unsaturated public
`active_gate_index / 6`; all other inputs are camera, IMU/actuator, and action
history. No oracle action is executed or included in the policy input. Native
geometry may shape offline reward and randomized curricula only.

Require at least 50,000 end-to-end agent steps/s, exact legal-boundary and
progress audits, finite logs, and at least five checkpoints. This proves fast
deployable-Puffer training and authorizes only teacher-free deterministic
offline screens. It sends zero FlightSim packets and grants no replay, shadow,
live, or Submission authority.
