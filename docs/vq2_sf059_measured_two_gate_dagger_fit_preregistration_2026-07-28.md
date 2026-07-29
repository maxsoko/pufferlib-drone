# VQ2-SF059 phase-balanced four-source DAgger fit — 2026-07-28

Tag: `vq2_sf059_measured_two_gate_dagger_fit_001`

Fine-tune the one SF056 actor on a virtual agent-axis union of SF049, SF052,
SF055, and SF058. Interleave four 64-agent sources into 256 logical agents and
reserve the final 32 logical agents, eight per source, for validation. Preserve
genuine episode starts and lengths; use SF049 steps `0..1471` and every valid
DAgger row.

Keep the architecture and trainable boundary unchanged. Use seed `42059`, 12
epochs, batch 8, chunks 64, AdamW `3e-5`, weight decay `1e-5`, gradient clip
`1.0`, action weights `[4,4,4,1]`, and smoothness `1e-4`. Because the three
phase-zero archives now contain substantially more rows than the first
on-policy phase-1 archive, use phase-zero loss weight `1` and Gate-2 loss weight
`2`. This is record-count balancing, not a runtime phase action selector.

Require phase-zero and Gate-2 weighted validation MSE `<=0.02` and every
channel MSE `<=0.05`. A numerical pass permits only a fresh SF060 exact
teacher-free two-gate screen. Write no labels, send zero FlightSim packets,
never access N712, and do not authorize shadow, bounded flight, or Submission.
