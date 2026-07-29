# VQ2-SF061 source-balanced phase-1 continuation — 2026-07-28

Tag: `vq2_sf061_source_balanced_dagger_fit_001`

Continue the one SF059 actor on eight virtual source groups: one each of SF049,
SF052, and SF055, plus five repetitions of SF058. Repetition is sampling weight
only; each logical agent still starts at a genuine episode boundary. This makes
the long clean and short on-policy Gate-2 record counts comparable. Interleave
the eight groups into 512 logical agents and reserve the last 64, eight per
group, for validation.

Use seed `42061`, 12 epochs, batch 8, chunks 64, AdamW `1e-5`, weight decay
`1e-5`, gradient clip `1.0`, phase-zero/Gate-2 loss weights `1/2`, action
weights `[4,4,4,1]`, smoothness `1e-4`, and the unchanged 779,204-parameter
trainable boundary. Select an admitted epoch first, otherwise lowest aggregate
partition score.

After training, separately evaluate held-out SF058 agents 56--63. Require the
normal aggregate admission plus SF058-only phase-zero and Gate-2 weighted MSE
each `<=0.01` and every SF058 channel MSE `<=0.05`. Only that combined pass may
permit a fresh SF062 exact teacher-free screen. Send zero FlightSim packets,
never access N712, and do not authorize shadow, bounded flight, or Submission.
