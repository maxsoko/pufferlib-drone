# VQ2 C003R handoff phase result — 2026-07-28

C003R is a valid rejection of the fixture configuration, not a policy screen.
Both source-locked recurrent replays are exact, but both reports record zero
plant steps and `selected_suffix_steps=[0,0]`.

The native visual environment's training-only `ordered_gate_phase` is defined
as `current_gate / num_gates`. C003R configured a two-gate segment, so its
initial Gate-2 phase was `1/2`. The composite correctly interpreted that as
already beyond the next `2/6` transition and stopped before inference. The
reports' apparent `128/128` `next_gate_passed` is therefore invalid and must
never be used as policy evidence.

Frozen invalid artifacts:

- true-range report SHA-256
  `48437679375fd4d0a9d7c121fbe421f012ab6b8ad63abb8e6c56cc483b184705`;
- alias report SHA-256
  `726b814144718735c7a52c039f5375c8ed67b61499a26de04f81d195ae92b398`;
- both empty traces SHA-256
  `9f56fcf32e0f69e78b000af2733893f3d9b4ca129af82d4ae033244d49bba818`.

The only contract correction is to retain six as the checkpoint's offline
phase-feature denominator while still stopping at the next ordered transition.
The feature scale is not an official gate-count assumption or finish rule.
Also set the observable denominator/one-hot fields explicitly and assert the
reset phase is exactly `1/6` in the focused native test. No policy, state,
plant, alias, action, or admission threshold changes.

Use new C004 tags. C003R sent zero FlightSim packets and executed zero plant,
teacher, student-update, or Submission action.
