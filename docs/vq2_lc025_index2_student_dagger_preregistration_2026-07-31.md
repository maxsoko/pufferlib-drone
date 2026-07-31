# VQ2 LC025 phase-2 student-state recollection — 2026-07-31

LC024 selects a 1% interpolation toward the first phase-2 student-state fit.
It raises paired mean progress from `2.875` to `3.25` gates and lowers crash
rate from `0.1875` to `0.15625`, but 16 of 32 runs still stop at index 2.
Recollect phase-2 labels on states owned by the updated recurrent Puffer.

Run 256 full-start 24-gate native episodes at fresh seed `431250`, 32 threads,
and at most 4,000 steps. The deterministic Puffer owns every plant action.
Query the offline native oracle only while held public progress equals 2.
Require at least 100,000 finite in-envelope phase-2 records, all episodes,
exact public-status/action-history transport, and zero teacher plant actions.

LC025 sends zero FlightSim packets and authorizes only a separately locked
phase-2 fit and interpolation bracket. It grants no live or Submission authority.
