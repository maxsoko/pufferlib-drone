# VQ2-SF056 exact measured three-source DAgger fit — 2026-07-28

Tag: `vq2_sf056_measured_two_gate_dagger_fit_001`

Fine-tune the single SF053 actor on a virtual agent-axis union of SF049's clean
oracle prefix, SF052's low-crash states, and SF055's near-miss states. Interleave
the 64 agents from each source into 192 logical agents; never concatenate
episodes in time. Reserve the final 18 logical agents, six from each source,
for validation. Use SF049 steps `0..1471` and every valid DAgger step.

Keep the SF053 training boundary and loss unchanged: train the encoder, one
256-wide GRU, phase embedding, joint action head, and joint phase residual;
freeze log standard deviation. Use seed `42056`, 16 epochs, batch 8, chunks 64,
AdamW `3e-5`, weight decay `1e-5`, gradient clip `1.0`, phase-zero/Gate-2
weights `2/1`, action weights `[4,4,4,1]`, and smoothness `1e-4`.

Require both partition weighted MSE values `<=0.02` and every channel MSE
`<=0.05`. A numerical pass permits only a fresh SF057 exact teacher-free
two-gate screen. Write no labels, send zero FlightSim packets, never access
N712, and do not authorize shadow, bounded flight, or Submission.
