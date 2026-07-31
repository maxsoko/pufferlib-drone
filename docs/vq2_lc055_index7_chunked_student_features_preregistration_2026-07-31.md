# VQ2 LC055 chunked Phase-7 collection — 2026-07-31

LC052's stable 1,024-agent run needed 408.914 seconds to obtain 10,000 phase-7
records from only six trajectories. LC053 and LC054 both show that this is not
enough trajectory diversity for a held-out fit. LC051's unchunked 2,048-agent
attempt reached no phase-7 state because changing the CUDA inference batch
changed the recurrent policy trajectory.

LC055 separates native simulation width from policy numerical width. Advance
2,048 native environments but evaluate the frozen LC048 Puffer in two ordered,
independent 1,024-row chunks at every step. Preserve recurrent row order and
state exactly. Stop deterministically after at least 20,000 phase-7 records;
require at least ten distinct query agents, at least one terminal agent, exact
phase/action transport, no hard native fault, and student ownership of every
plant action. Compare records, query trajectories, and wall time with LC052 to
measure useful-data throughput; do not infer success merely from GPU load.

LC055 is offline training-data collection. It sends zero FlightSim packets and
grants no live or Submission authority.
