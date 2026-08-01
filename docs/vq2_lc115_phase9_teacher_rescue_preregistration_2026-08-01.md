# LC115 exact phase-9 training-oracle rescue preregistration

LC105 is the frozen whole-Puffer offline frontier. In its serialization-exact
256-agent fixed-24 screen, one trajectory per duplicated 128-seed cohort
reaches raw progress 9, but no trajectory reaches raw progress 10. LC111's
isolated phase-9 residual endpoint preserves that entrant at every safe scale
tested through `1.0`, yet LC113 and LC114 remain 0/128 at raw progress 10.
Another policy fit is not justified until the training-only alignment oracle
demonstrates that this exact phase-9 state is controllably recoverable.

Run one paired native diagnostic with the saved LC105 actor, 256 agents, two
identical 128-seed cohorts, seed `432050`, 24 proxy gates, 32 CPU threads, CUDA
inference, and a 12,000-step bound. Advance the single recurrent Puffer actor
over all 256 rows. The first cohort is pure Puffer. In the second cohort only,
replace the complete four-action Puffer output with the training-only alignment
oracle exactly while the held public progress is 9. No teacher action is
permitted below phase 9, in the baseline cohort, or after raw progress 10; each
trajectory resolves on raw progress 10 or a native terminal.

The intervention rescues phase 9 only if the paired candidate creates at least
one raw-10 pass, loses no baseline pass, adds no pre-target terminal, executes
every action exactly, has no nonfinite/envelope/progress fault, and records no
teacher action outside candidate phase 9. A rescue authorizes one source-locked
phase-9 intervention collection and policy-only distillation. It does not
admit a Puffer checkpoint, authorize FlightSim, or authorize Submission. If
the oracle does not rescue the entrant, reject this phase-9 oracle family and
retain LC105 while diagnosing the teacher/control target offline.
