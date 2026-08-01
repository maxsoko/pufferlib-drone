# LC095 dense late-phase local-teacher collection preregistration

LC094 is the full-batch serialization-exact offline frontier, but it averages
only `3.421875` gates and reaches raw index 9 once in 128 runs. Waiting for
ordinary full starts therefore starves every later indexed Puffer head. Tiny
output-bias changes are also batch-numerically brittle.

Collect one dense training-only corpus with 512 gate-local episodes on the
fixed 24-gate proxy. Uniformly sample start phases 6 through 23, offsets 2--5 m,
seed 431950, 32 threads, and a 2,048-step horizon. The native alignment teacher
owns every plant action after the local reset; teacher blend remains zero and
the LC094 Puffer is evaluated only to produce its legal recurrent features and
pre-tanh baseline. Record teacher labels for every held phase 6--23.

Admission requires all 512 episodes complete, at least 150,000 total records,
at least 2,000 records per late phase, exact one-per-agent initial public-phase
jumps and otherwise ordered transport, crash rate at most 5%, finite in-envelope
labels, and exact executed action history. This is training-only teacher use:
it authorizes one multi-head offline fit and no teacher action at evaluation or
runtime. FlightSim and Submission remain forbidden.
