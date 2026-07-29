# VQ2 C003 handoff execution failure — 2026-07-28

C003A and C003B stopped before creating a native plant or output directory.
The evaluator's prefix replay guard detected a maximum SF066 action difference
above `5e-5` and raised before the measured handoff.

The source-locked diagnosis is numerical call-path mismatch:

- N294 whole-chunk replay error: `0.0`;
- N294 deployment-step replay error: `0.0`;
- SF066 whole-sequence GRU replay error: `0.00012729317`;
- SF066 deployment-step GRU replay error: `0.0`.

C002R generated the trace by calling `forward_step` once per public
observation, exactly as deployment does. C003 incorrectly warmed the same GRU
with one `forward_sequence` call. Both are mathematically equivalent, but the
CUDA GRU kernels do not reproduce the same floating-point path closely enough
for the source-lock threshold.

The only correction is to replay the 208 SF066 prefix observations through
`forward_step` and preserve its final state. No policy, observation, plant,
handoff, alias, threshold, or action path changes. Use new `vq2_c003r_*` tags;
never reuse the invalid C003A/B tags. No FlightSim packet, teacher action,
student update, native plant step, or Submission action occurred.
