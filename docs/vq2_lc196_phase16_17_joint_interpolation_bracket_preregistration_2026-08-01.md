# VQ2 LC196 phase-16/17 joint interpolation bracket preregistration

LC195 showed that the full LC194 phase-16 update regresses all trajectories
from raw index 17 to 16. LC196 therefore preserves the source direction and
tests a source-locked joint grid: phase-16 scales `0`, `0.0025`, `0.005`, and
`0.01`; phase-17 scales `0.25`, `0.5`, and `1.0`, plus the exact LC189
baseline. Thirteen independent recurrent adapter Puffers execute paired
32-agent seed-15 groups with zero teacher blend.

A source checkpoint is written only if a candidate reaches raw index 18 in
32/32 trajectories, gains all 32 paired outcomes over the exact raw-17
baseline, has zero losses and transport faults, and does not increase
pre-target terminals. Selection minimizes phase-16 scale first, then phase-17
scale. Any selection requires independent confirmation and remains offline;
FlightSim and VQ2 Submission are forbidden.
