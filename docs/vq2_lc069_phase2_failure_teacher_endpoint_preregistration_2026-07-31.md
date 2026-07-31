# LC069 phase-2 failure-teacher endpoint preregistration

LC068 showed that the failure classifier direction is causal but not
full-course promotable: it reduced crashes from 13/64 to 7/64 while losing one
Gate-3 pass and 0.015625 mean gates. LC069 therefore uses terminal outcomes
more directly and does not reuse the rejected action direction.

Use the source-locked 431,450 LC028 phase-2 records and LC062's frozen
phase-2 64-feature Puffer MLP. Split whole agents, stratified by terminal
outcome, into deterministic 80/20 train and validation sets. Give each outcome
class equal total weight and each trajectory within a class equal weight.

For successful trajectories, the regression target is LC062's existing
phase-2 residual, exactly anchoring the retained behavior. For failed
trajectories, the target is the training-only teacher action converted to
pre-tanh residual space. Fit only a proposed phase-2 output weight and bias;
the encoder, recurrent state, base action head, observation ABI, and every
other phase remain frozen. Select a ridge endpoint on held agents only.

Admission requires finite values, at least 1.10x held-failure teacher-action
MSE improvement, and no more than 0.0025 held-success action MSE drift from
LC062. An admitted endpoint authorizes only a small, paired, teacher-free
Gate-3 interpolation screen. It is not itself promoted and grants no live or
Submission authority. No teacher value or outcome label is a runtime input.

