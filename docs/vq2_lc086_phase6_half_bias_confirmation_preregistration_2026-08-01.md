# LC086 phase-6 half-bias confirmation preregistration

LC085 selected scale `0.5` of the successful-action direction, the exact
phase-6 bias `[+0.005,-0.025,+0.0125,0]`. On its 32 paired seeds it produced
two raw-index-7 passes versus one for LC073, with one paired gain, zero losses,
and one fewer pre-target terminal. Scale `1.0` had the same pass count but a
later mean crossing, so the smaller-displacement tie-break selected `0.5`.

Run 128 fresh paired baselines and 128 exact half-bias checkpoints in one
256-agent vector, seed 431860, 32 native threads, target raw index 7, and
12,000 steps. Every action is the deterministic mean of a complete recurrent
Puffer checkpoint. Teacher actions, outcomes, native state, and analytic
overrides are absent.

Confirm only with at least two additional raw-index-7 passes, positive paired
net gains, no added pre-target terminal, exact transport, and zero envelope
violations. Confirmation authorizes one new full-course screen only. Failure
rejects the half-bias without unchanged retry. No live or Submission authority
is granted.
