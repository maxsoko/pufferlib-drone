# VQ2-SF060 exact measured two-gate teacher-free screen — 2026-07-28

Tag: `vq2_sf060_measured_two_gate_teacher_free_512`

Evaluate frozen SF059 once under the unchanged exact measured screen with seed
`42060`, 512 instances, zero randomization/teacher blend, six rendered gates,
`0.75 m` apertures, one 4 Hz held public phase scalar, deterministic mean of
one actor's complete four-channel action, and a 2048-step cap.

Pass only on 512/512 ordered held phase `2/6`, zero premature terminal, phase
fault, nonfinite/envelope action, or action-delivery error above `5e-5`. A pass
permits a fresh perturbation ladder only. A failure rejects SF059 and forbids
an unchanged retry. Send zero FlightSim packets, never access N712, and do not
authorize shadow, bounded flight, or Submission.
