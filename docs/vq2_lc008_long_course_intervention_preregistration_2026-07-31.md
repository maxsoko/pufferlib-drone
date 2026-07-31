# VQ2 LC008 long-course intervention features — 2026-07-31

Run one offline training-only collection tagged
`vq2_lc008_long_course_intervention_features_001`: 128 fixed full-start
24-gate episodes, seed `431080`, 32 native threads, and at most `38,400`
steps. Warm the converted VG071 recurrent Puffer continuously. Its complete
deterministic action owns public index 0. Beginning only when the held 4 Hz
public progress reaches index 1, the offline alignment oracle owns the plant
and labels compact records through index 23.

The feature file contains only Puffer hidden state, Puffer base pre-tanh
action, teacher action, public index, agent, step, and terminal flag. No
privileged pose or gate geometry is stored as an actor input. Require at least
95% Gate-1-warmed full-course success, zero crash and hard native fault,
coverage of every head 1 through 23, at least two million exact plant/label
records, exact action history, exact unsaturated `active_gate_index / 6`
sample-and-hold behavior, and finite in-envelope values.

This corpus may authorize one offline recurrent Puffer residual fit. Classical
actions are training-only and may never reach replay, shadow, FlightSim, or
Submission. LC008 sends zero FlightSim packets and grants no live authority.
