# LC099 late-phase safe-alpha full-course preregistration

LC095 proved a dense late-phase Puffer training cycle: 975,450 legal records
across proxy phases 6--23 were collected in 17.109 seconds. LC096 fit all 18
phase heads in 17.544 seconds, and LC098's teacher-free local screen promoted
the crash-free alpha-0.15 interpolation. It increased mean gate-local advance
from `0.3541667` to `0.4270833`, increased one-gate passes from 26 to 32, and
removed the baseline cohort's single crash. That local result does not yet
prove benefit from a normal course start.

Run one paired full-start screen on the fixed 24-gate offline proxy. Load the
LC094 and LC098 checkpoints independently. Each actor receives the same full
256-agent input batch and owns a full recurrent state; group 0 executes only
LC094 vectors and group 1 only LC098 vectors. Post-forward surgery, teacher
actions, and analytic overrides are forbidden. Use 128 paired seeds per group,
seed 431990, 32 CPU threads, CUDA inference, and at most 12,000 native steps.

Promote LC098 only if it gains at least `1/128` mean ordered gates, adds one
raw-index-7 pass, does not increase crash rate, does not reduce maximum raw
index, and passes exact transport checks. The 24-gate course is an offline
proxy for the approximately 20-plus-gate official VQ2 event observed by the
user; only a nonnegative official finish time can prove an official lap.
LC099 cannot authorize FlightSim control or Submission.
