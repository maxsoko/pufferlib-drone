# LC063 phase-2 multiaxis bias preregistration

LC062 retains a whole-Puffer phase-2 pitch delta `-0.0025`; it improves paired
24-gate mean progress by `0.0625`, gains one Gate-3 pass, and reduces crash by
one episode. Phase 2 nevertheless remains the largest early stop.

LC063 uses the 17-second milestone layout to test eight paired groups of 32 at
new seed `431630`. Every listed value is an additional phase-2 pre-tanh output
bias on LC062: zero; pitch `-0.00125`; pitch `+0.00125`; roll `-0.0025`;
thrust `+0.0025`; thrust `-0.0025`; yaw `+0.0025`; and combined pitch
`-0.00125`, roll `-0.0025`, thrust `+0.0025`. The course remains 24 gates and
the milestone remains raw index 3 within 3,500 steps.

The plant action is always the exactly equivalent whole-Puffer output. Native
raw progress scores the offline milestone only. Advance a candidate to a
different-seed confirmation only if it gains a Gate-3 pass without increasing
pre-Gate-3 terminals and all transport checks pass. This run sends no
FlightSim packets and grants no live or Submission authority.
