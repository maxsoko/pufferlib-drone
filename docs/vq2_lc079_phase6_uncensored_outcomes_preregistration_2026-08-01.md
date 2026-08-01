# LC079 phase-6 uncensored outcome collection preregistration

LC077 rejected decoder interpolation and LC078 rejected the measured constant
phase-6 bias family. LC075 reached phase 6 quickly but stopped as soon as its
20,000-record target was met; 16 of its 18 queried trajectories were still
active, so their final records cannot be used as success anchors.

Run one fresh deterministic batch of 256 LC073-owned trajectories on the fixed
24-gate proxy, with seed 431790, 32 native threads, and the ordinary 12,000-step
horizon. Query the training-only alignment teacher only while held public index
is exactly 6, but do not stop on record count. The recurrent LC073 Puffer owns
every plant action and all 256 trajectories must reach a terminal state.

Admission requires at least 20,000 phase-6 records from at least eight agents,
zero censored queried trajectories, at least two phase-6 successes and two
phase-6 failures, exact action/history/progress transport, finite in-envelope
labels, and zero hard native faults. A success is a completed trajectory whose
last phase-6 record is nonterminal; a failure terminates on its last phase-6
record. This definition is valid only because the whole batch completes.

An admitted corpus authorizes one source-locked success-anchored phase-6
decoder fit. It grants no live or Submission authority. No teacher value,
native state, or outcome label may become a runtime input.
