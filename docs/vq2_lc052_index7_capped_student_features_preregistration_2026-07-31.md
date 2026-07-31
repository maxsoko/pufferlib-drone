# VQ2 LC052 1024-agent capped phase-7 corpus — 2026-07-31

LC051 is rejected without unchanged retry after the 2,048-agent batch produced
zero phase-7 bytes beyond the expected arrival window. Repeat the changed
collection layout at 1,024 agents, the batch size that produced LC045's useful
late-phase states. Keep the exact LC048 recurrent Puffer, 24-gate course,
phase-7-only oracle query, seed `431520`, 128 threads, and 12,000-step ceiling.

Stop after the first complete vector step reaching at least 10,000 records;
require fewer than 11,024 records, at least two distinct query agents, some
terminal evidence, exact phase/action transport, finite in-envelope labels,
and zero teacher plant action. The smaller corpus is an admission probe for an
isolated phase-7 fit, not a robustness claim.

LC052 is training-only, sends zero FlightSim packets, and grants no live or
Submission authority.
