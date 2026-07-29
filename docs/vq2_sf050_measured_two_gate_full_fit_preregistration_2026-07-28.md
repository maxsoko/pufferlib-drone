# VQ2-SF050 measured two-gate full fit — 2026-07-28

Tag: `vq2_sf050_measured_two_gate_full_fit_001`

Fine-tune the single SF045 phase-aware actor on admitted SF049 measured-course
oracle prefixes. Use exactly steps `0..1471`: held phase is zero through step
495, `1/6` through step 1471, and changes to `2/6` at step 1472. Thus no later
gate state enters this two-gate fit.

Use agents 0--55 for training and 56--63 for validation. Train the encoder,
one 256-wide GRU, joint four-action head, phase embedding, and joint phase
residual; freeze only log standard deviation. Use seed `42050`, 16 epochs,
batch 8, chunks 64, AdamW `0.00003`, weight decay `0.00001`, gradient clip
`1.0`, phase-zero loss weight `2`, Gate-2 loss weight `1`, action weights
`[4,4,4,1]`, and smoothness `0.0001`.

Select an admitted epoch first, otherwise the lowest summed validation score.
Require phase-zero weighted MSE `<=0.03`, Gate-2 weighted MSE `<=0.02`, and
every channel MSE `<=0.05`. A numerical pass permits only a fresh exact
measured two-gate native screen. Send zero FlightSim packets, do not access
N712, and do not authorize shadow, bounded flight, or Submission.

