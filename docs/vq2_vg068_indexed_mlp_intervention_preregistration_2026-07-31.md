# VQ2 VG068 indexed nonlinear intervention-feature fit — 2026-07-31

Run one epoch-resumable offline fit tagged
`vq2_vg068_indexed_mlp_intervention_features_001`. VG067 closes the linear
hidden-feature residual line: its smallest scale reduces count-5 finishes and
increases crashes. VG063 nevertheless provides 4,666,436 warmed feature rows
from `512/512` collision-free completed 5--12-gate teacher interventions.

Keep the complete VG033 encoder, recurrent core, decoder, phase embedding, and
standard deviation exact. Add one learned 256->64 tanh->4 residual MLP per
public index. Initialize every output layer/bias to exact zero so behavior is
base-exact. Fit indices 1--11 only by direct post-tanh four-action MSE; outputs
for index 0 and 12--16 must remain zero. This remains one recurrent full-output
Puffer actor with no runtime teacher, blend, clip, fallback, or controller.

Use seed `429203`, ten epochs, 65,536-row chunks, AdamW `2e-3`, weight decay
`1e-5`, gradient clip 1, and group-stratified held agents where
`(agent_index//8)%8==0`. Average losses across phases present in each chunk.
Persist exact optimizer/model/best/history state after every epoch.

Numerically admit only if held phase-balanced MSE improves at least 2x,
phases 1--3 each improve at least 2x, no phase regresses at all, all base
parameters are exact, inactive outputs are zero, values are finite, and total
trainable L2 is at most 256. Admission authorizes only a teacher-free scale
diagnostic; no rollout or live authority exists before it.

Bind the pushed commit, VG033, VG063 features/report/admission, VG067 rejection,
goal, nonlinear model, trainer, runner, tests, runtime, and safety fields.
Frozen new hashes are filled before launch:

- model: `94e07edcb36bd1aaa22501a69801cc6a8d6d1389c5f4ae85c9894ddb0b246b08`
- trainer: `a8b53ab908e2b33c6dea3b8d310d9428c1075f24586482d2a488aa7147b1586f`
- runner: `a2b99c30d2a46f3fa446be5d83daae837837f9d793f66917bd64247d8d35dba9`
- tests: `fb9b17fd688c7193579cfb1c379b940787f4fa2a196bcdc76da9b34a444dbe84`

FlightSim, shadow, Training, Submission, teacher runtime actions, and sealed
test access are zero. VQ2 Submission remains forbidden.
