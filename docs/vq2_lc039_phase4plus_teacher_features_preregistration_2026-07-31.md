# VQ2 LC039 corrected phase-4+ teacher corpus — 2026-07-31

LC038 reached phase 17 and then stopped before an artifact because its wrapper
left the reusable collector's phase counter at the legacy 16-gate bound. LC039
changes only that counter allocation to the already established 32-phase
long-course cap and uses a new tag and seed `431390`; it does not reuse LC038's
partial staging data.

All remaining LC038 controls and thresholds are unchanged: 128 full-start
24-gate episodes, the exact LC037 recurrent Puffer through held phases 0--3,
the training-only oracle from held phase 4 onward, 32,000 steps, and labels for
phases 4--23. Require at least 15% phase-4 reach, 10% completed teacher
continuations, crash at most 20%, 400,000 total labels, and 1,000 labels per
late phase, with exact action/progress transport.

LC039 sends zero FlightSim packets and grants no live or Submission authority.
