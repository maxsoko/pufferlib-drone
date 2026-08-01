# LC120 outcome-balanced phase-6--9 distillation preregistration

LC119 admits 60,082 exact LC117 milestone records from 19 phase-6--9
intervention trajectories: 4 reach raw progress 10 and 15 fail. Fit the complete
two-layer indexed Puffer residual head independently at phases 6, 7, 8, and 9,
starting from frozen LC105. The deployed ABI and all non-target parameters stay
unchanged.

Use a stratified per-phase agent split, equal total weight for successful and
failed trajectory classes, equal weight per trajectory within a class, teacher
actions as the target for both classes, 512 Adam steps per phase, batch 4096,
learning rate `.001`, parameter anchor `.001`, gradient clip `1`, and target
clip `.999`. Admit a phase only if validation teacher-action MSE improves by at
least `1.2x` overall and `1.1x` in each outcome class, with finite parameters
and parameter delta L2 at most 64.

Admission creates an offline endpoint only. It authorizes one teacher-free
paired scale bracket of complete saved-form Puffer checkpoints at raw progress
10. It does not authorize FlightSim or Submission.
