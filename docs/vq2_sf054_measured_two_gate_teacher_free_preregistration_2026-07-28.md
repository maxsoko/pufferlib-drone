# VQ2-SF054 exact measured two-gate teacher-free screen — 2026-07-28

Tag: `vq2_sf054_measured_two_gate_teacher_free_512`

Evaluate frozen, numerically admitted SF053 once on `512` parallel native
instances using seed `42054` and the unchanged SF051 contract: exact measured
Gate-1/Gate-2 geometry, six rendered gates, official `0.75 m` apertures, zero
start/gate/camera/plant randomization, and a hard `2048`-step bound. Stop an
instance only when the `4 Hz` status-rate-held public phase reaches `2/6`.
This is an offline two-gate milestone, not an official finish claim.

The one SF053 recurrent Puffer actor emits the deterministic mean of all four
actions at every active step from exactly 4,118 legal values plus the one held
public phase scalar. Teacher blend, controller, labels, optimizer updates,
overrides, selectors, and fallbacks are absent.

Require `512/512` ordered two-gate milestones, zero premature terminals, phase
decreases/skips/off-tick changes, nonfinite or envelope-invalid outputs, and
delivered-action error above `5e-5`. A pass permits only a fresh measured
perturbation ladder. A failure rejects SF053 for promotion and permits a new
policy-state DAgger iteration, never an unchanged retry. Send zero FlightSim
packets, never access N712, and do not authorize shadow, bounded flight, or
Submission.
