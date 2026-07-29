# VQ2 C007 measured visual suffix DAgger fit result — 2026-07-28

C007 is numerically admitted and authorizes only the paired C008 offline
closed-loop screen. It does not authorize FlightSim.

- selected epoch: `11` of `16`;
- optimizer updates: `12,076` total;
- Gate-2 held-out weighted MSE: `0.0006934876815094308`;
- phase-zero held-out weighted MSE: `0.0007933826391403096`;
- all five source-specific audits pass the preregistered weighted and
  per-channel thresholds;
- source-specific C006 true/alias weighted MSE:
  `0.00045469300134697846/0.001885162945690628`;
- FlightSim/Submission packets: `0/0`.

Frozen artifacts:

- checkpoint SHA-256
  `393f5de6f9dd97ae81d855b23d77c96e97926507e7c2b89611714df3d60a24c5`;
- report SHA-256
  `bfc99b3d5177c5e83d9d46653480c26c66e05eebbc8d1d1bbc960bacd77c6e49`.

The numerical fit is not flight evidence. Run the child exactly once on each
of the frozen C005 true-range and 16-step live-alias handoffs. Reject it if
either distribution is not perfect and fault-free.
