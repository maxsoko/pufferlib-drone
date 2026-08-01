# LC122 phase-6/8 composite scale screen preregistration

LC121 rejects the phase-6-only endpoint at scales `.1,.3,1`. Scale `.3` is the
retained non-destructive component: it preserves one raw-9 trajectory, creates
two raw-8 trajectories, and has exact transport. LC120 independently admits
its phase-8 head with `1.408x` overall validation improvement.

Screen complete saved-form Puffer candidates consisting of fixed phase-6 scale
`.3` plus phase-8 scales `.1,.3,1`, against byte-exact LC105. Preserve every
other tensor. Use the same 128 duplicated seeds per candidate, seed `432050`,
24 proxy gates, 12,000-step raw-10 target, and baseline-plus-candidate 256-row
actor execution contract as LC121.

Select only a composite that creates at least one raw-10 pass with no paired
loss or added pre-target terminal and exact transport. Selection authorizes an
independent offline confirmation only; FlightSim and Submission remain
forbidden.
