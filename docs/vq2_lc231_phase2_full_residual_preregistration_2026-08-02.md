# VQ2 LC231 deployment-context phase-2 residual preregistration

LC230's custom admission predicate expected every distinct control seed to fail,
but its actual paired result was more informative: controls passed 4/8, the
training-only oracle passed 8/8, and it rescued all four failed pairs with zero
losses. The base collector's causal rescue predicate passed. All 46,804 records
are finite, in-envelope, phase-2 records with teacher targets for both control
and intervention states. LC229 separately proves that deployment seed 287 fails
8/8 when repeated under exact NumPy batch-1 execution.

LC231 source-locks the immutable LC230 report and feature binary and performs
one complete phase-2 residual-head fit. It may update only the four indexed
phase-2 residual tensors; the encoder, recurrent Puffer state, action head,
every other indexed phase, adapter, and phase-16--23 action sequence remain
frozen. The optimization is 512 Adam steps, 4,096 rows per step, learning rate
0.003, anchor coefficient 0.001, and equal-agent train/validation weighting.
Admission requires at least 1.5x validation teacher-MSE improvement, a finite
endpoint, and phase-row parameter delta no greater than 64.

An admitted endpoint may be exported losslessly to a hash-bound NumPy archive.
It receives no FlightSim authority. The next mandatory test is teacher-free
native closed loop using independent exact batch-1 NumPy callable instances,
first through Gate 3 and then through all 24 proxy gates. VQ2 Submission remains
forbidden.
