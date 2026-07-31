# VQ2 LC018 native Gate-1 low-variance PPO — 2026-07-31

LC017 rejected all first-pass PPO checkpoints at Gate 1. Restart one legal
native Puffer from fresh weights with exploration `log_std=-2`, zero entropy
bonus, and a fixed one-gate full-start curriculum. Train 10,000,384 agent
steps using LC016's 1,024-agent, 16-buffer, 128-thread, horizon-16 native
layout. Randomize Gate-1 aperture across 0.75--2.0 m and retain randomized
legal camera geometry; do not use local starts or teacher actions.

The actor input boundary remains exact: camera/IMU/actuator/action history plus
public progress only, with the 34-value native training tail zero. Save every
~2M steps. This rung tests whether low-variance native PPO can establish a
deterministic Gate-1 base before expanding toward the observed ~20+ gate
course. It sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
