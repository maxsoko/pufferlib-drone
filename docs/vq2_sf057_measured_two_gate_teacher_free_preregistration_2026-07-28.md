# VQ2-SF057 exact measured two-gate teacher-free screen — 2026-07-28

Tag: `vq2_sf057_measured_two_gate_teacher_free_512`

Run frozen SF056 once under the unchanged SF054 screen contract with fresh seed
`42057`: 512 exact measured-course native instances, six rendered gates,
`0.75 m` apertures, no randomization, 4 Hz held public phase, deterministic
mean actions, teacher blend exactly zero, and a 2048-step cap.

Require 512/512 ordered held-phase `2/6` milestones, no premature terminal,
phase fault, nonfinite/envelope action, or delivered-action error above `5e-5`.
The same one recurrent actor must emit all four channels. A pass permits only a
fresh perturbation ladder; a failure rejects SF056 for promotion and forbids an
unchanged retry. This screen sends zero FlightSim packets, never accesses N712,
and cannot authorize shadow, bounded flight, or Submission.
