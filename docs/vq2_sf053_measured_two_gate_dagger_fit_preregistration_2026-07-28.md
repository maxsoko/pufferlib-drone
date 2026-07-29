# VQ2-SF053 exact measured two-source DAgger fit — 2026-07-28

Tag: `vq2_sf053_measured_two_gate_dagger_fit_001`

Fine-tune the single frozen-parent SF050 phase actor on a virtual, source-locked
union of the admitted SF049 oracle prefix and SF052 policy-state DAgger data.
Do not copy or concatenate episodes in time. Interleave the 64 agents from each
source into 128 logical agents so every recurrent state begins at a genuine
episode start; reserve the final 16 logical agents (8 from each source) for
validation.

Use SF049 steps `0..1471` and every valid SF052 step. Train the encoder, one
256-wide GRU, phase embedding, joint four-action head, and joint phase residual;
freeze only log standard deviation. Use seed `42053`, 16 epochs, agent batch 8,
sequence chunks 64, AdamW `3e-5`, weight decay `1e-5`, gradient clip `1.0`,
phase-zero loss weight `2`, Gate-2 loss weight `1`, action weights `[4,4,4,1]`,
and smoothness `1e-4`.

Select an admitted epoch first, otherwise lowest summed validation score.
Require phase-zero and Gate-2 weighted MSE each `<=0.02` and every action-channel
MSE `<=0.05`. A numerical pass permits only a fresh SF054 exact measured
teacher-free two-gate screen. Write no new labels, send zero FlightSim packets,
never access N712, and do not authorize shadow, bounded flight, or Submission.
