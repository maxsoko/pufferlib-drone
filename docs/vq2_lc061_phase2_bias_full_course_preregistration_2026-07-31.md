# LC061 phase-2 bias full-course preregistration

LC059 and LC060 independently favored the same whole-Puffer phase-2 pitch
output-bias delta `-0.0025`. LC061 tests whether that local Gate-3 gain improves
downstream progress rather than promoting from a milestone.

Run one command-free native vector with two identical-seed groups of 64,
seed `431610`, 24 ordered gates, 12,000 steps, 32 OpenMP threads, held 4 Hz
public progress, and one batched recurrent actor. Group 0 is frozen LC048;
Group 1 is LC048 with only phase-2 indexed output bias changed before tanh.
The evaluator's vectorized action is exactly representable by the saved
whole-Puffer checkpoint. No teacher, analytic runtime action, FlightSim packet,
or Submission authority exists.

Promote the candidate offline only if mean gates improve by at least `0.05`,
Gate-3 passes improve by at least one, crash rate does not increase, maximum
raw index does not regress, and both paired groups pass transport checks. Even
promotion here authorizes only continued offline diagnosis, not a live run.
