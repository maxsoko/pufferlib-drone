# LC089 phase-7 parity-bias confirmation preregistration

LC088 selected scale `0.5` of the mirrored phase-7 direction, the exact bias
`[+0.005,+0.025,+0.0125,0]`. On its 32 paired seeds it produced two
raw-index-8 passes versus one for LC087, with one paired gain, zero losses, and
one fewer pre-target terminal. Scales `0.75`, `1.0`, and `1.5` tied the pass
count but changed more parameters; scales `2` and `3` lost the baseline pass.

Run 128 fresh paired LC087 baselines and 128 exact half-scale phase-7
checkpoints in one 256-agent vector, seed 431890, 32 native threads, target raw
index 8, and 12,000 steps. Every action is the deterministic mean of a complete
recurrent Puffer checkpoint. Teacher actions, outcomes, native runtime state,
and analytic overrides are absent.

Confirm only with at least two additional raw-index-8 passes, positive paired
net gains, no added pre-target terminal, exact transport, and zero envelope
violations. Confirmation authorizes one new full-course screen only. Failure
rejects parity transfer without unchanged retry. No live or Submission
authority is granted.
