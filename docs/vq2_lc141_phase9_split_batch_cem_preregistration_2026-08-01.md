# VQ2 LC141 split-batch phase-9 CEM preregistration

LC140 did not exercise its search: one 512-row actor changed LC105 recurrent
numerics and every identical seed-15 row stopped at raw index 6, so phase 9 was
never queried.  Preserve the same 512-environment seed-15 vector and the same
bounded, teacher-free CEM search, but execute two independent complete 256-row
LC105 actors.  This is the source-measured actor batch shape used by the LC139
screen in which seed index 15 reaches raw index 9.

Require byte-identical initial native states, all 512 rows querying held public
phase 9, exact transport, and at least one raw-10 pass before saving a candidate.
Only the selected candidate's indexed phase-9 residual output-bias row may
change.  Stop at the first successful generation, up to four generations.

This is a corrected execution contract, not an unchanged LC140 retry.  It has
teacher-free deterministic screen authority only.  It sends no FlightSim
packet and does not authorize Submission.  The offline proxy has 24 gates;
direct simulator inspection indicates approximately 20 official gates or more,
and only official finish status proves an official lap.
