# LC075 phase-6 horizon-corrected feature preregistration

LC074 is terminally rejected: all 256 episodes were transport-clean, but no
trajectory reached held index 6 inside the 6,000-step bound, so it produced
zero records and no checkpoint. LC073's full-course evidence reaches index 6
under the ordinary 12,000-step horizon. LC075 changes only that insufficient
collection horizon plus a fresh tag and seed.

Run 256 LC073-owned trajectories, 32 native threads, seed 431750, and the
ordinary 12,000-step bound. Query only held public index 6 and stop at the
first tick containing at least 20,000 but fewer than 20,256 records from at
least eight agents. Preserve every LC074 transport, legality, teacher-query,
and no-live predicate.

An admitted corpus authorizes one source-locked phase-6 Puffer decoder fit.
No live or Submission authority is granted, and LC074 must never be relabeled
or retried unchanged.

