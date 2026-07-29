# VQ2-SF068 exact two-gate teacher-free PPO — 2026-07-28

Tag: `vq2_sf068_exact_two_gate_ppo_001`

Continue frozen SF066 with recurrent PPO on the exact measured first two gates.
Use `num_gates=2` only to terminate training immediately after Gate 2; convert
the native training phase back to official `active_gate_index / 6` before it
reaches the actor. The actor remains exactly 4,119 legal/public inputs and four
complete CTBR outputs. It receives no pose, gate geometry, teacher action,
reward component, value, or critic feature.

Use a separate training-only feed-forward critic on the 34-value normalized
native privileged tail. Native geometry may shape reward only. Pin every
teacher blend/imitation term to zero. Use seed `42068`, 128 agents, 48 updates,
64-step recurrent rollouts, 16-agent minibatches, two PPO epochs, actor learning
rate `3e-6`, critic learning rate `3e-4`, clipped ratio `0.1`, exploration std
`0.06`, gamma `0.997`, lambda `0.95`, and reward scale `0.01`. Reset recurrent
state across terminal and visual reset-only boundaries.

Evaluate the deterministic mean on 128 exact two-gate episodes before training
and every four updates. Select only by deterministic native results, ordered by
success, crash, gates passed, then terminal radial error. Admit training only
if the selected checkpoint is `128/128` successful, zero crash/miss/timeout/
out-of-order, and exactly two gates passed. Any later official-format screen is
separate and source-locked.

Send zero FlightSim packets, never access N712, and do not authorize a shadow,
bounded flight, or Submission.
