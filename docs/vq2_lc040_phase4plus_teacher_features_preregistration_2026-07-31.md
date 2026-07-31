# VQ2 LC040 corrected phase-4 teacher boundary — 2026-07-31

LC039 produced 3,166,365 finite, balanced labels for phases 4--23 and completed
`128/128` teacher continuations with zero crashes, but it is rejected because
its teacher plant selector retained the reusable module's phase-1 default.
The query boundary remained phase 4, leaving 465,152 unrecorded teacher plant
actions at phases 1--3. LC040 passes the configured phase boundary explicitly,
adds a focused regression test, uses a new tag and seed `431400`, and does not
train from LC039.

All other LC039 controls and thresholds are unchanged: 128 full-start 24-gate
episodes, LC037 Puffer plant actions through held phases 0--3, training-only
oracle plant actions from phase 4, 32,000 steps, at least 400,000 total labels,
and at least 1,000 labels for every phase 4--23. Teacher plant actions must now
equal recorded labels exactly.

LC040 sends zero FlightSim packets and grants no live or Submission authority.
