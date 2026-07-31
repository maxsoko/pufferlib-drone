# VQ2 LC028 third phase-2 student-state collection — 2026-07-31

LC027 selects a 5% interpolation from the LC024 parent toward the LC026 fit.
It raises paired mean progress from `3.25` to `3.50`, doubles index-6 reaches
to six, and lowers crash rate from `0.15625` to `0.0625`; 13 of 32 runs still
stop at index 2. Recollect only phase-2 labels on states owned by this Puffer.

Run 256 full-start 24-gate episodes at fresh seed `431280`, 32 threads, and at
most 3,500 steps. The deterministic Puffer owns every plant action. The oracle
is queried only at held public phase 2 and executes zero plant actions. Require
at least 100,000 finite labels, exact transport, and all episodes.

LC028 sends zero FlightSim packets and grants no live or Submission authority.
