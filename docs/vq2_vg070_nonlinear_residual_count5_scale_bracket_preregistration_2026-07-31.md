# VQ2 VG070 nonlinear residual count-5 scale bracket — 2026-07-31

Run one resumable offline bracket tagged
`vq2_vg070_nonlinear_residual_count5_scale_bracket_001`. VG069 exactly replays
VG068's training history and passes every numerical gate after independently
selecting its phase heads. This experiment is the single rollout diagnostic
authorized by that admission.

Use one complete `VQ2IndexedPhaseMLPResidualActor` with the frozen VG033 base
and VG069 nonlinear heads. For alpha zero, keep VG069's fitted hidden features
but set every residual output layer and bias to exact zero; this is behaviorally
identical to VG033. For positive alphas, scale only each fitted output layer and
bias by alpha. No action mixing or runtime teacher exists: the resulting one
Puffer actor emits every four-action vector directly.

Screen alphas `0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0` on 64 fresh
native count-5 episodes at offset `208`, seed `429204`, four threads, and the
admitted long-course horizon of 23,040 steps. A candidate qualifies only if
transport is clean, Gate 1 does not regress, crashes do not exceed alpha zero,
and completed five-gate episodes strictly increase. Select by finishes, fewer
crashes, downstream reach, and then smaller alpha.

A pass authorizes only a separately preregistered independent multi-offset
screen. A failure rejects this fitted intervention line. Bind the pushed
commit, parent, VG069 checkpoint/report/admission, VG068 rejection, goal,
evaluator, runner, tests, native code/binding/config, runtime, and all safety
fields. Frozen new hashes are filled before launch:

- evaluator: `54be8f04cfe1fe4fb60cd018a4479df95ee36dff26fd27b985bf2dcda2d585b6`
- runner: `ea4f6aed048772510b97aea6ecb9ca7e5238b0848b1c773314ceffb3c51acb77`
- tests: `9e5f65d2ef7984f032881cc45f97ebc037dcf4dc7e5372618bc4e5264a6e7bb4`

FlightSim, Windows shadow, VQ2 Training, VQ2 Submission, teacher actions, and
sealed test access are zero. VQ2 Submission remains forbidden.
