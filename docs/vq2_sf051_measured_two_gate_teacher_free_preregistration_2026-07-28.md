# VQ2-SF051 exact measured two-gate teacher-free screen — 2026-07-28

Tag: `vq2_sf051_measured_two_gate_teacher_free_512`

Evaluate frozen SF050 once on `512` parallel native instances from the exact
measured full start and Gate-1/Gate-2 geometry. Use seed `42051`, the official
`0.75 m` aperture, zero start/gate/camera/plant randomization, and a hard
`2048`-step bound. Keep six gates rendered, but stop each instance only after
the status-rate-held native public equivalent reaches `active_gate_index / 6 =
2/6`. This is an offline two-gate milestone, not an official finish claim.

The same `VQ2PhaseResidualActor` checkpoint must emit the deterministic mean
of the complete four-action vector at every active step. Feed it only the
4,118-value legal observation plus one `4 Hz` sample-and-held normalized gate
index. Teacher action blend, native teacher controllers, action overrides,
checkpoint selection, and analytic fallback are exactly zero/absent.

Pass only if all `512/512` instances expose held phase `2/6`, every instance
passes Gate 1 before Gate 2, no phase decreases or skips, held phase changes
only on a status tick, no instance terminates before the milestone, all actor
and delivered actions are finite and within the normalized envelope, and the
delivered-action error is at most `5e-5`. Source-lock the SF050 checkpoint and
report. Write no labels and perform no updates.

A pass permits only a separately preregistered measured-transition
perturbation ladder. A failure rejects SF050 for teacher-free promotion and
permits a fresh policy-driven exact-course DAgger collection; it does not
permit an unchanged SF051 retry. Send zero FlightSim packets, never access the
consumed N712 sealed test, and do not authorize shadow, bounded flight, or
Submission.
