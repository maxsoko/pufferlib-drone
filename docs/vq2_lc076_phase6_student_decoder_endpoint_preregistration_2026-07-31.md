# LC076 phase-6 student-decoder endpoint preregistration

LC075 admits 20,007 training-only phase-6 labels from 18 trajectories owned by
the retained LC073 Puffer. LC076 fits a low-cost decoder endpoint before any
closed-loop rollout.

Freeze LC073's recurrent model, base action head, phase-6 256-to-64 encoder,
and every non-target parameter. Split whole query agents deterministically
80/20. Give each trajectory equal total weight. Regress teacher actions in
pre-tanh residual space onto the frozen 64 legal recurrent features using the
ridge grid `1e-6` through `1e-1`; select solely by held-agent teacher-action
MSE.

Admission requires finite parameters, endpoint L2 no greater than 64, and at
least 1.10x held-agent action-MSE improvement over the exact LC073 phase-6
decoder. The endpoint is not directly deployable. Admission authorizes only a
small teacher-free interpolation milestone screen. Teacher actions and native
state remain training-only, and no live or Submission authority is granted.

