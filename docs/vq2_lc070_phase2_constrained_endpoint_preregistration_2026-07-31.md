# LC070 phase-2 constrained endpoint preregistration

LC069's direct outcome-conditioned endpoint improved held-failure teacher
action MSE by 2.73x but was rejected because its 0.1376 held-success drift was
well above the 0.0025 cap. The direction contains useful failure correction;
its unconstrained magnitude does not.

Keep the LC069 ridge solution fixed and evaluate source-locked interpolation
fractions `0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30` from LC062
toward it on the same held-agent partition. Select the nonzero fraction with
minimum held-failure teacher-action MSE among candidates whose held-success
action drift is no more than 0.0025. Require at least 1.03x failure improvement
over LC062 and finite parameters.

This is an offline numerical constraint step, not a closed-loop admission.
An admitted scaled endpoint authorizes one fresh-seed, teacher-free Gate-3
interpolation screen only. Runtime remains the single recurrent Puffer policy;
outcomes and teacher actions are absent. No live or Submission authority is
granted.

