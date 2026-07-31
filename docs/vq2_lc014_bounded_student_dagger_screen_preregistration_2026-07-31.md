# VQ2 LC014 bounded student DAgger screen — 2026-07-31

Run one fresh-seed teacher-free diagnostic of LC013: 32 full-start 24-gate
episodes, seed `431140`, 32 native threads, deterministic CUDA mean actions,
and at most 12,000 steps. The recurrent state starts at zero and advances
continuously. Every plant action is the complete LC013 Puffer output using held
4 Hz `active_gate_index / 6` progress.

Require exact action delivery, finite in-envelope actions, ordered monotonic
public progress, and zero privileged values or teacher actions at the actor
input. Compare mean and maximum gates against LC011's rejected `1.09375` mean
and maximum index 2. Only a transport-clean behavioral improvement can
authorize the next student-state DAgger rung or a larger screen.

LC014 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
